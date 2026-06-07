import sys
import rclpy
import numpy as np
from typing import Optional
from rclpy.lifecycle import Node, Publisher, State, TransitionCallbackReturn
from rclpy.timer import Timer
from rclpy.executors import ExternalShutdownException
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from std_msgs.msg import Header
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

        self.declare_parameters(
            namespace='',
            parameters=[
                ('frame_id',    'tof_frame'),
                ('transport',   'uart'),
                ('serial_port', '/dev/sen0628'),
                ('i2c_addr',    51),          # 0x33
                ('timer_period', 0.1),
                ('osc_enable',  False),
                ('osc_ip',      '127.0.0.1'),
                ('osc_port',    8000),
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
        """Read one frame and return (point_buf, res_rows, res_cols) or None."""
        try:
            dist = self.sensor.read_frame()
        except Exception as e:
            self.get_logger().error(f'Sensor read error: {e}')
            return None

        if dist is None:
            return None

        res_r, res_c = dist.shape
        per_px_r = np.deg2rad(45) / res_r
        per_px_c = np.deg2rad(45) / res_c

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

        return buf, res_r, res_c

    def publish_pcl(self):
        if self.pcl_pub is None or not self.pcl_pub.is_activated:
            return
        result = self.read_sensor()
        if result is None:
            return
        buf, res_r, res_c = result
        point_size = 3 * 4
        pc_msg = PointCloud2(
            header=Header(
                stamp=Time().to_msg(),
                frame_id=self.get_parameter('frame_id').value),
            height=res_r,
            width=res_c,
            fields=[
                PointField(name='x', offset=0,  datatype=PointField.FLOAT32, count=1),
                PointField(name='y', offset=4,  datatype=PointField.FLOAT32, count=1),
                PointField(name='z', offset=8,  datatype=PointField.FLOAT32, count=1)],
            is_bigendian=False,
            is_dense=False,
            point_step=point_size,
            row_step=point_size * res_c,
            data=buf.tobytes()
        )
        self.pcl_pub.publish(pc_msg)
        self.publish_osc(buf)

    def on_configure(self, state: State) -> TransitionCallbackReturn:
        transport   = self.get_parameter('transport').value
        serial_port = self.get_parameter('serial_port').value
        i2c_addr    = self.get_parameter('i2c_addr').value

        try:
            if transport == 'i2c':
                self.sensor = Sen0628I2c(i2c_addr)
                self.get_logger().info(f'Using I2C transport, addr=0x{i2c_addr:02X}')
                # I2C requires mode configuration
                if not self.sensor.set_Ranging_Mode(8):
                    self.get_logger().error('set_Ranging_Mode() failed')
                    return TransitionCallbackReturn.FAILURE
            else:
                self.sensor = Sen0628Uart(serial_port)
                self.get_logger().info(f'Using UART transport on {serial_port}')

            self.get_logger().info('Waiting for first sensor frame...')
            dist = self.sensor.read_frame(timeout=5.0)
            if dist is None:
                self.get_logger().error('No data from sensor within 5 s')
                return TransitionCallbackReturn.FAILURE
            self.get_logger().info(
                f'Sensor ready: {dist.shape[0]}x{dist.shape[1]} matrix')
        except Exception as e:
            self.get_logger().error(f'Failed to open sensor: {e}')
            return TransitionCallbackReturn.FAILURE

        self.get_logger().info('Configured: Inactive')
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: State) -> TransitionCallbackReturn:
        try:
            self.pcl_pub = self.create_lifecycle_publisher(
                PointCloud2, 'pointcloud', qos_profile=qos_profile_sensor_data)
            self.timer = self.create_timer(
                self.get_parameter('timer_period').value, self.publish_pcl)
            self.setup_osc()
            self.get_logger().info('Activated')
        except Exception as e:
            self.get_logger().error(f'Activation failed: {e}')
        return super().on_activate(state)

    def on_deactivate(self, state: State) -> TransitionCallbackReturn:
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
        if self.sensor is not None:
            try:
                self.sensor.close()
            except Exception:
                pass
            self.sensor = None
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
