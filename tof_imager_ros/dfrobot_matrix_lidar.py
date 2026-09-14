# DFRobot SEN0628 matrix LiDAR drivers
#
# UART: device streams ASCII lines continuously:
#   y<row>:<v0>,<v1>,...,<vN>,\r\n   (distance mm;  4000 = out of range)
#   s<row>:<v0>,<v1>,...,<vN>,\r\n   (signal kcps/SPAD; 0 = no target)
#   s-lines are emitted by custom firmware only; old DFRobot firmware omits them.
#
# read_frame() always returns (dist, signal):
#   dist   — float32 (rows, cols) mm, NaN where out-of-range
#   signal — float32 (rows, cols) kcps/SPAD, NaN where no target,
#             or None if the sensor does not emit s-lines
#
# I2C: binary command-response protocol (DFRobot vendor library, MIT License)

import time
import serial
import smbus
import numpy as np


# ── UART (ASCII streaming) ────────────────────────────────────────────────────

class Sen0628Uart:
    def __init__(self, port='/dev/sen0628', baudrate=115200):
        # write_timeout is not optional here: the V1.3 firmware never drains
        # its USB-CDC input endpoint, so a write with no timeout blocks
        # forever and takes bringup down with it. Measured, twice.
        self.ser = serial.Serial(port, baudrate, timeout=1, write_timeout=2)
        time.sleep(0.1)
        self.ser.reset_input_buffer()
        # The y0 line that ended the previous frame is the first line of the
        # next one. Holding it here instead of dropping it is what lets
        # consecutive calls read consecutive frames — see read_frame.
        self._pending = None

    # Vendor command frame, from DFRobot_MatrixLidar: head 0x55, then the
    # argument count big/little, then the command, then its arguments. The
    # library writes it a byte at a time with a 1 ms gap, which is reproduced
    # here — the device is streaming while it listens, and a burst is not what
    # its parser was tested against.
    CMD_SETMODE = 1
    HEAD = 0x55

    def set_ranging_mode(self, matrix, settle=6.0):
        """Switch between the 4x4 and 8x8 matrix. matrix is 4 or 8.

        The sensor restarts ranging afterwards, so this drops whatever is in
        flight and waits before the caller reads again — the vendor library
        sleeps 5 s at this point for the same reason.
        """
        if matrix not in (4, 8):
            raise ValueError(f'matrix must be 4 or 8, got {matrix}')
        pkt = bytes([self.HEAD, 0x00, 0x05, self.CMD_SETMODE, 0, 0, 0, matrix])
        try:
            for b in pkt:
                self.ser.write(bytes([b]))
                time.sleep(0.001)
        except serial.SerialTimeoutException:
            # Expected on SEN0628-V1.3. What the device actually does, tested
            # on a freshly replugged sensor: it accepts one write (8 bytes,
            # out_waiting back to 0), does not act on it — still 8x8 after —
            # and from then on its USB-CDC input endpoint stalls, so every
            # later write times out until it is physically replugged.
            #
            # Not fatal. The sensor keeps streaming in whatever mode it is
            # already in, and a ranging mode we could not select is no reason
            # to refuse to run. The caller warns; the robot carries on.
            return False
        # Deliberately no flush(): it waits for the device to drain and has no
        # timeout, which is the call that hung the robot when this was written.
        time.sleep(settle)
        self.ser.reset_input_buffer()
        self._pending = None          # stale half-frame from the old shape
        return True

    def read_frame(self, timeout=2.0):
        """Return (dist, signal) for one complete frame.

        dist:   float32 ndarray (rows, cols) in mm, NaN = out-of-range
        signal: float32 ndarray (rows, cols) in kcps/SPAD, NaN = no target,
                or None when the sensor doesn't emit s-lines (old firmware)
        Returns (None, None) on timeout.
        """
        deadline = time.monotonic() + timeout
        y_rows = {}
        s_rows = {}

        # Sync to a y0 line, starting from the one the previous call stopped on.
        # Without that hand-off this consumed a whole frame and then threw away
        # the y0 that terminated it, so the next call had to wait for the frame
        # after — halving the output rate. Measured on the robot: the device
        # streams 9.83 Hz and the topic published exactly 5.00 Hz.
        if self._pending is not None:
            y_rows[0] = self._parse_vals(self._pending)
            self._pending = None
        else:
            while time.monotonic() < deadline:
                line = self._readline()
                if line and line.startswith('y0:'):
                    y_rows[0] = self._parse_vals(line)
                    break
            else:
                return None, None

        # Collect remaining y/s rows until next y0 (= start of next frame)
        while time.monotonic() < deadline:
            line = self._readline()
            if not line or ':' not in line:
                continue
            prefix = line[0]
            if prefix not in ('y', 's'):
                continue
            try:
                row = int(line[1:line.index(':')])
            except ValueError:
                continue
            if prefix == 'y':
                if row == 0:
                    self._pending = line     # first line of the next frame
                    break
                y_rows[row] = self._parse_vals(line)
            else:
                s_rows[row] = self._parse_vals(line)

        if not y_rows:
            return None, None

        n_rows = len(y_rows)
        n_cols = max(len(v) for v in y_rows.values())

        dist = np.full((n_rows, n_cols), np.nan, dtype=np.float32)
        for r, vals in y_rows.items():
            for c, v in enumerate(vals[:n_cols]):
                if v < 4000:
                    dist[r, c] = float(v)

        signal = None
        if s_rows:
            signal = np.full((n_rows, n_cols), np.nan, dtype=np.float32)
            for r, vals in s_rows.items():
                for c, v in enumerate(vals[:n_cols]):
                    if v > 0:
                        signal[r, c] = float(v)

        return dist, signal

    def close(self):
        if self.ser.is_open:
            self.ser.close()

    def _readline(self):
        # A vanished device raises here (SerialException / OSError) and must
        # reach the caller: swallowing it turned an unplugged cable into an
        # endless, silent, CPU-burning read loop on a dead descriptor.
        return self.ser.readline().decode('ascii', errors='ignore').strip()

    @staticmethod
    def _parse_vals(line):
        _, _, vals_str = line.partition(':')
        vals = []
        for tok in vals_str.rstrip(',').split(','):
            tok = tok.strip()
            if tok:
                try:
                    vals.append(int(tok))
                except ValueError:
                    pass
        return vals


# ── I2C (binary command-response, DFRobot vendor protocol) ───────────────────

class _DFRobotBase:
    CMD_SETMODE   = 1
    CMD_ALLData   = 2
    DEBUG_TIMEOUT = 8
    STATUS_OK     = 0x53
    STATUS_FAIL   = 0x63
    ERR_NONE      = 0x00
    ERR_TIMEOUT   = 0x04

    INDEX_ARGS_H  = 0
    INDEX_ARGS_L  = 1
    INDEX_CMD     = 2
    INDEX_ERR     = 0
    INDEX_STATUS  = 1
    INDEX_RES_CMD = 2
    INDEX_LEN_L   = 3
    INDEX_LEN_H   = 4
    INDEX_DATA    = 5

    def set_Ranging_Mode(self, matrix):
        pkt = [0, 5, self.CMD_SETMODE, 0, 0, 0, matrix]
        self._send(pkt)
        time.sleep(0.1)
        resp = self._recv(self.CMD_SETMODE)
        if (len(resp) >= 5 and resp[self.INDEX_ERR] == self.ERR_NONE
                and resp[self.INDEX_STATUS] == self.STATUS_OK):
            time.sleep(5)
            return True
        return False

    def read_frame(self, timeout=2.0):
        """Return (dist, None) — I2C transport has no signal data."""
        pkt = [0, 1, self.CMD_ALLData]
        self._send(pkt)
        time.sleep(0.1)
        resp = self._recv(self.CMD_ALLData)
        if (len(resp) < 5 or resp[self.INDEX_ERR] != self.ERR_NONE
                or resp[self.INDEX_STATUS] != self.STATUS_OK):
            return None, None
        length = resp[self.INDEX_LEN_L] | (resp[self.INDEX_LEN_H] << 8)
        raw = resp[self.INDEX_DATA:]
        if len(raw) < length:
            return None, None
        n = length // 2
        side = int(n ** 0.5)
        if side * side != n:
            return None, None
        vals = [raw[i * 2] | (raw[i * 2 + 1] << 8) for i in range(n)]
        dist = np.array(vals, dtype=np.float32).reshape(side, side)
        dist[dist == 0] = np.nan
        return dist, None

    def close(self):
        pass

    def _recv(self, cmd):
        t0 = time.time()
        while time.time() - t0 < self.DEBUG_TIMEOUT:
            b = self._read(1)
            if not b:
                continue
            s = b[0]
            if s in (self.STATUS_OK, self.STATUS_FAIL):
                c = self._read(1)
                if not c or c[0] != cmd:
                    return [self.ERR_TIMEOUT]
                ll = self._read(2)
                if not ll or len(ll) < 2:
                    return [self.ERR_TIMEOUT]
                length = (ll[1] << 2) | ll[0]
                if length > 128:
                    return [self.ERR_TIMEOUT]
                result = [self.ERR_NONE, s, c[0], ll[0], ll[1]]
                if length:
                    result += self._read(length)
                return result
            time.sleep(0.05)
        return [self.ERR_TIMEOUT]


class Sen0628I2c(_DFRobotBase):
    def __init__(self, addr=0x33, bus=1):
        self._addr = addr
        self._bus = smbus.SMBus(bus)

    def _send(self, pkt):
        try:
            self._bus.write_i2c_block_data(self._addr, 0x55, pkt)
        except Exception:
            pass

    def _read(self, length):
        result = []
        for _ in range(length):
            try:
                result.append(self._bus.read_byte(self._addr))
            except Exception:
                result.append(0)
        return result
