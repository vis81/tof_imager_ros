import os
import sys
import rclpy
import numpy as np
from typing import Optional
from rclpy.lifecycle import Node, Publisher, State, TransitionCallbackReturn
from rclpy.timer import Timer
from rclpy.executors import ExternalShutdownException
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from rclpy.time import Time
from std_msgs.msg import Header
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from sensor_msgs.msg import PointCloud2, PointField
from tof_imager_ros.dfrobot_matrix_lidar import Sen0628Uart, Sen0628I2c

try:
    from pythonosc import udp_client
    _OSC_AVAILABLE = True
except ImportError:
    _OSC_AVAILABLE = False


class ToFImagerPublisher(Node):
    def __init__(self, node_name='tof_imager'):
        super().__init__(node_name)
        self.pcl_pub: Optional[Publisher] = None
        self.timer: Optional[Timer] = None
        self.osc_client = None
        self.sensor = None
        # Device loss and recovery are handled here, not by a restart: on a
        # read error (the cable came out) the port is closed and a slow
        # timer waits for the device path to come back, reopens it and
        # resumes. See _lose_sensor / _try_reopen.
        self.retry_timer: Optional[Timer] = None
        self.empty_frames = 0
        self.diag_pub = self.create_publisher(DiagnosticArray, '/diagnostics', 5)

        self.declare_parameters(
            namespace='',
            parameters=[
                ('frame_id',         'tof_frame'),
                ('transport',        'uart'),
                ('serial_port',      '/dev/sen0628'),
                ('i2c_addr',         51),          # 0x33
                # 8 or 4: the 8x8 or 4x4 ranging matrix. 8 is the useful one
                # here -- the vertical rows are what separates a low obstacle
                # from the floor, which is the whole reason this sensor is on
                # the robot. 4x4 is offered because the sensor's ranging budget
                # is per-zone, so fewer zones can be sampled faster.
                ('ranging_mode',     8),
                ('timer_period',     0.1),
                ('min_signal_kcps',  0),            # 0 = disabled; drop zones below this signal
                # SEN0628 ranging angle, from the DFRobot wiki: 60 deg
                # horizontally, 60 deg vertically (90 diagonal). This was
                # hardcoded at 45, which placed every point at the wrong
                # bearing -- see the projection below.
                ('fov_deg',          60.0),
                ('osc_enable',       False),
                ('osc_ip',           '127.0.0.1'),
                ('osc_port',         8000),
            ]
        )
        self.get_logger().info('Initialized')

    def setup_osc(self):
        if not self.get_parameter('osc_enable').value:
            return
        if not _OSC_AVAILABLE:
            self.get_logger().warn('python-osc not installed; OSC disabled')
            return
        osc_ip   = self.get_parameter('osc_ip').value
        osc_port = self.get_parameter('osc_port').value
        try:
            self.osc_client = udp_client.SimpleUDPClient(osc_ip, osc_port)
            self.get_logger().info(f'OSC initialized: {osc_ip}:{osc_port}')
        except Exception as e:
            self.get_logger().error(f'OSC init failed: {e}')

    def publish_osc(self, buf):
        if not self.get_parameter('osc_enable').value or self.osc_client is None:
            return
        try:
            self.osc_client.send_message('/tx', buf[:, :, 0].flatten().tolist())
            self.osc_client.send_message('/ty', buf[:, :, 1].flatten().tolist())
            self.osc_client.send_message('/tz', buf[:, :, 2].flatten().tolist())
        except Exception as e:
            self.get_logger().error(f'OSC send error: {e}')

    def read_sensor(self):
        """Read one frame and return (point_buf, signal_flat_valid, res_rows, res_cols) or None.

        signal_flat_valid is a 1-D float32 array aligned to the valid xyz points,
        or None when the sensor doesn't provide signal data (old firmware / I2C).
        """
        if self.sensor is None:
            return None
        try:
            dist, signal = self.sensor.read_frame()
        except Exception as e:
            self._lose_sensor(f'read error: {e}')
            return None

        if dist is None:
            # Present but silent (a stalled endpoint, a device that will be
            # gone from /dev a moment later): a few seconds of that and it is
            # treated like a missing device - close, wait, reopen.
            self.empty_frames += 1
            if self.empty_frames >= 3:
                self._lose_sensor('no frames')
            return None
        self.empty_frames = 0

        min_signal = self.get_parameter('min_signal_kcps').value
        if min_signal > 0 and signal is not None:
            dist = dist.copy()
            dist[signal < min_signal] = np.nan

        res_r, res_c = dist.shape
        # The zone grid spans the sensor's full ranging angle, so the angular
        # pitch is fov/resolution. Hardcoding 45 here against a 60 deg sensor
        # scaled every bearing by 0.75 and pulled the whole cloud toward the
        # optical axis: measured on the robot, the floor came back 2.1 cm above
        # where it is, leaving only 0.9 cm of margin under the costmap's 3 cm
        # min_obstacle_height -- less than a carpet pile, which is why carpet
        # marked as an obstacle.
        fov = np.deg2rad(self.get_parameter('fov_deg').value)
        per_px_r = fov / res_r
        per_px_c = fov / res_c

        # Pre-compute per-pixel tangent offsets once per frame shape change.
        # The sensor reports Z-depth (distance along each zone's optical axis,
        # perpendicular to the sensor face), so we use perspective projection:
        #   x = d, y = d·tan(az), z = d·tan(el)
        # Using spherical projection (d·cos·sin) would make flat walls appear
        # concave because it treats depth as slant range.
        row_idx = np.arange(res_r, dtype=np.float32)
        col_idx = np.arange(res_c, dtype=np.float32)
        el_angles = ((res_r - 1) / 2.0 - row_idx) * per_px_r   # (res_r,)
        az_angles = ((res_c - 1) / 2.0 - col_idx) * per_px_c   # (res_c,)
        tan_el = np.tan(el_angles)[:, None]   # (res_r, 1)
        tan_az = np.tan(az_angles)[None, :]   # (1, res_c)

        d = dist / 1000.0   # mm → m,  shape (res_r, res_c)
        buf = np.full((res_r, res_c, 3), np.nan, dtype=np.float32)
        valid = ~np.isnan(d)
        buf[valid, 0] = d[valid]                      # x: forward (depth)
        buf[valid, 1] = (d * tan_az)[valid]            # y: left
        buf[valid, 2] = (d * tan_el)[valid]            # z: up

        # Extract signal values aligned to valid xyz points
        sig_valid = None
        if signal is not None:
            sig_valid = signal.reshape(-1)[valid.reshape(-1)].astype(np.float32)

        return buf, sig_valid, res_r, res_c

    def publish_pcl(self):
        if self.pcl_pub is None or not self.pcl_pub.is_activated:
            return
        result = self.read_sensor()
        if result is None:
            return
        # Stamp as soon as the frame is in hand, before the numpy work below.
        # read_sensor blocks on a complete UART frame, so this is already the
        # end of the measurement rather than the start of it — but it is the
        # closest honest time we have, and it is what the costmap needs to put
        # the returns where the robot was, not where it has since got to.
        stamp = self.get_clock().now().to_msg()
        buf, sig_valid, res_r, res_c = result

        # Flatten to a dense unorganized cloud — strips NaN (out-of-range) points.
        # Organized clouds (height>1) make PCL use OrganizedNeighbor search which
        # asserts on NaN; unorganized forces KdTree which handles sparse data.
        pts_all = buf.reshape(-1, 3)
        valid_mask = ~np.isnan(pts_all[:, 0])
        pts = pts_all[valid_mask]

        if sig_valid is not None:
            # xyzi layout (16 bytes/point): x, y, z, intensity
            point_data = np.column_stack([pts, sig_valid]).astype(np.float32)
            fields = [
                PointField(name='x',         offset=0,  datatype=PointField.FLOAT32, count=1),
                PointField(name='y',         offset=4,  datatype=PointField.FLOAT32, count=1),
                PointField(name='z',         offset=8,  datatype=PointField.FLOAT32, count=1),
                PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1),
            ]
            point_step = 16
        else:
            # xyz only — backward compatible with old firmware / I2C transport
            point_data = pts
            fields = [
                PointField(name='x', offset=0,  datatype=PointField.FLOAT32, count=1),
                PointField(name='y', offset=4,  datatype=PointField.FLOAT32, count=1),
                PointField(name='z', offset=8,  datatype=PointField.FLOAT32, count=1),
            ]
            point_step = 12

        pc_msg = PointCloud2(
            header=Header(
                # Not Time(): that default-constructs to zero, which tf2 reads
                # as "use the latest transform available". The cloud then gets
                # placed at wherever the robot is when the message is handled
                # instead of where it was when the frame was captured — about
                # 14 cm of error at 0.7 m/s, in the unsafe direction, since the
                # obstacle lands beyond its true position.
                stamp=stamp,
                frame_id=self.get_parameter('frame_id').value),
            height=1,
            width=len(pts),
            fields=fields,
            is_bigendian=False,
            is_dense=True,
            point_step=point_step,
            row_step=point_step * len(pts),
            data=point_data.tobytes()
        )
        self.pcl_pub.publish(pc_msg)
        self.publish_osc(buf)

    def on_configure(self, state: State) -> TransitionCallbackReturn:
        transport   = self.get_parameter('transport').value
        serial_port = self.get_parameter('serial_port').value
        i2c_addr    = self.get_parameter('i2c_addr').value
        mode        = int(self.get_parameter('ranging_mode').value)
        if mode not in (4, 8):
            self.get_logger().error(f'ranging_mode must be 4 or 8, got {mode}')
            return TransitionCallbackReturn.FAILURE

        if transport == 'i2c':
            try:
                self.sensor = Sen0628I2c(i2c_addr)
                self.get_logger().info(f'Using I2C transport, addr=0x{i2c_addr:02X}')
                # I2C requires mode configuration
                if not self.sensor.set_Ranging_Mode(mode):
                    self.get_logger().error('set_Ranging_Mode() failed')
                    return TransitionCallbackReturn.FAILURE
            except Exception as e:
                self.get_logger().error(f'Failed to open sensor: {e}')
                return TransitionCallbackReturn.FAILURE
        elif not self._open_sensor():
            # Not a failure: the sensor is unplugged (or not ready yet).
            # Configure succeeds, and the retry timer started on activation
            # opens it the moment it is there.
            self.get_logger().warning(
                f'{serial_port} is not there yet - will open it when it appears')

        self.get_logger().info('Configured: Inactive')
        return TransitionCallbackReturn.SUCCESS

    def _open_sensor(self) -> bool:
        """Open the UART sensor and wait for its first frame. False if the
        device is absent or silent; nothing is left half-open."""
        serial_port = self.get_parameter('serial_port').value
        mode = int(self.get_parameter('ranging_mode').value)
        if not os.path.exists(serial_port):
            return False
        try:
            self.sensor = Sen0628Uart(serial_port)
            self.get_logger().info(f'Using UART transport on {serial_port}')
            if mode != 8 and not self.sensor.set_ranging_mode(mode):
                self.get_logger().warning(
                    f'Could not select the {mode}x{mode} matrix: the '
                    'SEN0628-V1.3 firmware does not act on mode commands '
                    'over UART, and stalls its input endpoint after the '
                    'first one. Still streaming, in the mode it is already '
                    'in. Use transport: i2c to change it.')
            self.get_logger().info('Waiting for first sensor frame...')
            dist, signal = self.sensor.read_frame(timeout=5.0)
            if dist is None:
                self.get_logger().error('No data from sensor within 5 s')
                self._close_sensor()
                return False
            has_signal = signal is not None
            self.get_logger().info(
                f'Sensor ready: {dist.shape[0]}x{dist.shape[1]} matrix'
                f'{", signal data available" if has_signal else ""}')
        except Exception as e:
            self.get_logger().error(f'Failed to open sensor: {e}')
            self._close_sensor()
            return False
        self.empty_frames = 0
        self._diag(DiagnosticStatus.OK, f'streaming on {serial_port}')
        return True

    def _close_sensor(self):
        if self.sensor is not None:
            try:
                self.sensor.close()
            except Exception:
                pass
            self.sensor = None

    def _lose_sensor(self, why: str):
        """The device is gone or silent: close it and start waiting for it.
        The 50 Hz read timer is stopped meanwhile - polling a dead port at
        that rate was a fifth of a Pi core."""
        self.get_logger().warning(f'sensor lost ({why}); waiting for it to come back')
        self._close_sensor()
        self._diag(DiagnosticStatus.ERROR, f'sensor lost: {why}')
        if self.timer is not None:
            self.timer.cancel()
        if self.retry_timer is None:
            self.retry_timer = self.create_timer(1.0, self._try_reopen)

    def _try_reopen(self):
        if self.sensor is not None or not self._open_sensor():
            if self.sensor is None:
                self._diag(DiagnosticStatus.ERROR, 'sensor lost: waiting for the device')
            return
        self.get_logger().info('sensor is back')
        if self.retry_timer is not None:
            self.retry_timer.cancel()
            self.destroy_timer(self.retry_timer)
            self.retry_timer = None
        if self.timer is not None:
            self.timer.reset()

    def _diag(self, level, message: str):
        st = DiagnosticStatus(name='sen0628', hardware_id=str(self.get_parameter('serial_port').value),
                              level=level if isinstance(level, bytes) else bytes([level]),
                              message=message)
        st.values = [KeyValue(key='frame_id', value=str(self.get_parameter('frame_id').value))]
        msg = DiagnosticArray()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.status = [st]
        self.diag_pub.publish(msg)

    def on_activate(self, state: State) -> TransitionCallbackReturn:
        try:
            self.pcl_pub = self.create_lifecycle_publisher(
                PointCloud2, 'pointcloud',
                qos_profile=QoSProfile(
                    reliability=QoSReliabilityPolicy.RELIABLE,
                    history=QoSHistoryPolicy.KEEP_LAST,
                    depth=5))
            self.timer = self.create_timer(
                self.get_parameter('timer_period').value, self.publish_pcl)
            if self.sensor is None:
                self._lose_sensor('not open')
            self.setup_osc()
            self.get_logger().info('Activated')
        except Exception as e:
            self.get_logger().error(f'Activation failed: {e}')
        return super().on_activate(state)

    def on_deactivate(self, state: State) -> TransitionCallbackReturn:
        if self.retry_timer is not None:
            self.retry_timer.cancel()
            self.destroy_timer(self.retry_timer)
            self.retry_timer = None
        if self.timer is not None:
            self.timer.cancel()
            self.destroy_timer(self.timer)
            self.timer = None
        if self.pcl_pub is not None:
            self.destroy_publisher(self.pcl_pub)
            self.pcl_pub = None
        self.get_logger().info('Deactivated')
        return super().on_deactivate(state)

    def on_cleanup(self, state: State) -> TransitionCallbackReturn:
        self._close_sensor()
        self.get_logger().info('Cleaned up')
        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info('Shutting down')
        return TransitionCallbackReturn.SUCCESS


def main(args=None):
    rclpy.init(args=args)
    node = ToFImagerPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except ExternalShutdownException:
        sys.exit(1)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
