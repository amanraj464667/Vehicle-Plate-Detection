
# Simple speed estimation stub.
# Real system requires camera calibration to convert pixels -> meters.
# Here we provide a placeholder function that uses simple frame delta and a provided scale.

def estimate_speed(prev_center, curr_center, time_elapsed_seconds, meters_per_pixel=0.02):
    """Estimate speed in km/h.
    prev_center and curr_center are (x,y) pixel coordinates of the tracked object center.
    meters_per_pixel is a rough calibration parameter (user must tune depending on camera).
    """
    dx = curr_center[0] - prev_center[0]
    dy = curr_center[1] - prev_center[1]
    dist_pixels = (dx**2 + dy**2) ** 0.5
    dist_meters = dist_pixels * meters_per_pixel
    speed_m_s = dist_meters / max(time_elapsed_seconds, 1e-6)
    speed_kmh = speed_m_s * 3.6
    return speed_kmh
