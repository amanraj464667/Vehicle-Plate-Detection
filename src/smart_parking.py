import sqlite3
import datetime
import uuid
from pathlib import Path
import json

# Database path
PARKING_DB_PATH = Path(__file__).resolve().parents[1] / 'results' / 'smart_parking.db'

class SmartParkingSystem:
    def __init__(self):
        self.init_parking_db()
    
    def init_parking_db(self):
        """Initialize the smart parking database with all required tables"""
        conn = sqlite3.connect(str(PARKING_DB_PATH))
        cursor = conn.cursor()
        
        # Parking zones table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS parking_zones (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                zone_name TEXT UNIQUE NOT NULL,
                total_spots INTEGER NOT NULL,
                hourly_rate REAL NOT NULL,
                zone_type TEXT DEFAULT 'REGULAR' CHECK(zone_type IN ('REGULAR', 'VIP', 'DISABLED', 'ELECTRIC')),
                is_active INTEGER DEFAULT 1,
                created_date TEXT NOT NULL
            )
        ''')
        
        # Parking spots table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS parking_spots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                spot_number TEXT UNIQUE NOT NULL,
                zone_id INTEGER NOT NULL,
                is_occupied INTEGER DEFAULT 0,
                is_reserved INTEGER DEFAULT 0,
                spot_type TEXT DEFAULT 'REGULAR',
                last_updated TEXT,
                FOREIGN KEY (zone_id) REFERENCES parking_zones (id)
            )
        ''')
        
        # Vehicle entries table (active parking sessions)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS parking_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT UNIQUE NOT NULL,
                plate_number TEXT NOT NULL,
                spot_id INTEGER,
                zone_id INTEGER NOT NULL,
                entry_time TEXT NOT NULL,
                exit_time TEXT,
                entry_image_path TEXT,
                exit_image_path TEXT,
                vehicle_type TEXT DEFAULT 'CAR',
                parking_fee REAL DEFAULT 0,
                payment_status TEXT DEFAULT 'PENDING' CHECK(payment_status IN ('PENDING', 'PAID', 'OVERDUE')),
                is_active INTEGER DEFAULT 1,
                notes TEXT,
                FOREIGN KEY (spot_id) REFERENCES parking_spots (id),
                FOREIGN KEY (zone_id) REFERENCES parking_zones (id)
            )
        ''')
        
        # Parking violations table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS parking_violations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                plate_number TEXT NOT NULL,
                violation_type TEXT NOT NULL,
                violation_time TEXT NOT NULL,
                location TEXT,
                fine_amount REAL DEFAULT 0,
                image_path TEXT,
                status TEXT DEFAULT 'OPEN' CHECK(status IN ('OPEN', 'PAID', 'DISPUTED', 'WAIVED')),
                notes TEXT
            )
        ''')
        
        # Registered vehicles table (for monthly passes, VIP access, etc.)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS registered_vehicles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                plate_number TEXT UNIQUE NOT NULL,
                owner_name TEXT NOT NULL,
                owner_phone TEXT,
                owner_email TEXT,
                vehicle_type TEXT DEFAULT 'CAR',
                pass_type TEXT DEFAULT 'HOURLY' CHECK(pass_type IN ('HOURLY', 'DAILY', 'MONTHLY', 'VIP')),
                pass_expiry TEXT,
                is_active INTEGER DEFAULT 1,
                registered_date TEXT NOT NULL,
                notes TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
        
        # Initialize default zones if they don't exist
        self.setup_default_zones()
    
    def setup_default_zones(self):
        """Setup default parking zones"""
        default_zones = [
            ('Zone A - Regular', 50, 5.0, 'REGULAR'),
            ('Zone B - VIP', 10, 10.0, 'VIP'),
            ('Zone C - Disabled', 5, 0.0, 'DISABLED'),
            ('Zone D - Electric', 15, 3.0, 'ELECTRIC')
        ]
        
        conn = sqlite3.connect(str(PARKING_DB_PATH))
        cursor = conn.cursor()
        
        for zone_name, total_spots, hourly_rate, zone_type in default_zones:
            cursor.execute('''
                INSERT OR IGNORE INTO parking_zones (zone_name, total_spots, hourly_rate, zone_type, created_date)
                VALUES (?, ?, ?, ?, ?)
            ''', (zone_name, total_spots, hourly_rate, zone_type, datetime.datetime.now().isoformat()))
            
            # Create parking spots for this zone
            zone_id = cursor.lastrowid or self.get_zone_id_by_name(zone_name)
            if zone_id:
                for spot_num in range(1, total_spots + 1):
                    spot_number = f"{zone_type[0]}{spot_num:03d}"  # A001, B001, etc.
                    cursor.execute('''
                        INSERT OR IGNORE INTO parking_spots (spot_number, zone_id, spot_type, last_updated)
                        VALUES (?, ?, ?, ?)
                    ''', (spot_number, zone_id, zone_type, datetime.datetime.now().isoformat()))
        
        conn.commit()
        conn.close()
    
    def get_zone_id_by_name(self, zone_name):
        """Get zone ID by zone name"""
        conn = sqlite3.connect(str(PARKING_DB_PATH))
        cursor = conn.cursor()
        cursor.execute('SELECT id FROM parking_zones WHERE zone_name = ?', (zone_name,))
        result = cursor.fetchone()
        conn.close()
        return result[0] if result else None
    
    def vehicle_entry(self, plate_number, zone_name="Zone A - Regular", vehicle_type="CAR", image_path=None):
        """Handle vehicle entry to parking"""
        conn = sqlite3.connect(str(PARKING_DB_PATH))
        cursor = conn.cursor()
        
        # Check if vehicle is already parked
        cursor.execute('''
            SELECT id FROM parking_sessions 
            WHERE plate_number = ? AND is_active = 1
        ''', (plate_number.upper().replace(' ', ''),))
        
        if cursor.fetchone():
            conn.close()
            return False, "Vehicle is already parked"
        
        # Get zone information
        cursor.execute('SELECT id, hourly_rate FROM parking_zones WHERE zone_name = ? AND is_active = 1', (zone_name,))
        zone_info = cursor.fetchone()
        
        if not zone_info:
            conn.close()
            return False, "Parking zone not found"
        
        zone_id, hourly_rate = zone_info
        
        # Find available parking spot
        cursor.execute('''
            SELECT id, spot_number FROM parking_spots 
            WHERE zone_id = ? AND is_occupied = 0 AND is_reserved = 0 
            ORDER BY spot_number LIMIT 1
        ''', (zone_id,))
        
        spot_info = cursor.fetchone()
        if not spot_info:
            conn.close()
            return False, "No available parking spots in this zone"
        
        spot_id, spot_number = spot_info
        
        # Create parking session
        session_id = str(uuid.uuid4())[:8]
        entry_time = datetime.datetime.now().isoformat()
        
        cursor.execute('''
            INSERT INTO parking_sessions 
            (session_id, plate_number, spot_id, zone_id, entry_time, entry_image_path, vehicle_type)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (session_id, plate_number.upper().replace(' ', ''), spot_id, zone_id, entry_time, image_path, vehicle_type))
        
        # Mark spot as occupied
        cursor.execute('''
            UPDATE parking_spots 
            SET is_occupied = 1, last_updated = ? 
            WHERE id = ?
        ''', (entry_time, spot_id))
        
        conn.commit()
        conn.close()
        
        return True, f"Vehicle parked successfully in spot {spot_number}. Session ID: {session_id}"
    
    def vehicle_exit(self, plate_number, image_path=None):
        """Handle vehicle exit from parking"""
        conn = sqlite3.connect(str(PARKING_DB_PATH))
        cursor = conn.cursor()
        
        # Find active parking session
        cursor.execute('''
            SELECT ps.id, ps.session_id, ps.spot_id, ps.entry_time, ps.zone_id, pz.hourly_rate, psp.spot_number
            FROM parking_sessions ps
            JOIN parking_zones pz ON ps.zone_id = pz.id
            JOIN parking_spots psp ON ps.spot_id = psp.id
            WHERE ps.plate_number = ? AND ps.is_active = 1
        ''', (plate_number.upper().replace(' ', ''),))
        
        session_info = cursor.fetchone()
        if not session_info:
            conn.close()
            return False, "No active parking session found for this vehicle"
        
        session_id_db, session_id, spot_id, entry_time, zone_id, hourly_rate, spot_number = session_info
        
        # Calculate parking duration and fee
        exit_time = datetime.datetime.now()
        entry_datetime = datetime.datetime.fromisoformat(entry_time)
        duration_hours = max(1, (exit_time - entry_datetime).total_seconds() / 3600)  # Minimum 1 hour
        parking_fee = round(duration_hours * hourly_rate, 2)
        
        # Update parking session
        cursor.execute('''
            UPDATE parking_sessions 
            SET exit_time = ?, exit_image_path = ?, parking_fee = ?, is_active = 0 
            WHERE id = ?
        ''', (exit_time.isoformat(), image_path, parking_fee, session_id_db))
        
        # Mark spot as available
        cursor.execute('''
            UPDATE parking_spots 
            SET is_occupied = 0, last_updated = ? 
            WHERE id = ?
        ''', (exit_time.isoformat(), spot_id))
        
        conn.commit()
        conn.close()
        
        duration_str = f"{int(duration_hours)}h {int((duration_hours % 1) * 60)}m"
        return True, {
            "message": f"Vehicle exited successfully from spot {spot_number}",
            "session_id": session_id,
            "duration": duration_str,
            "parking_fee": parking_fee,
            "spot_number": spot_number
        }
    
    def get_parking_status(self):
        """Get overall parking status"""
        conn = sqlite3.connect(str(PARKING_DB_PATH))
        cursor = conn.cursor()
        
        # Get zone-wise occupancy
        cursor.execute('''
            SELECT pz.zone_name, pz.total_spots, 
                   COUNT(CASE WHEN ps.is_occupied = 1 THEN 1 END) as occupied_spots,
                   pz.hourly_rate, pz.zone_type
            FROM parking_zones pz
            LEFT JOIN parking_spots ps ON pz.id = ps.zone_id
            WHERE pz.is_active = 1
            GROUP BY pz.id, pz.zone_name, pz.total_spots, pz.hourly_rate, pz.zone_type
        ''')
        
        zones_status = cursor.fetchall()
        
        # Get active sessions count
        cursor.execute('SELECT COUNT(*) FROM parking_sessions WHERE is_active = 1')
        active_sessions = cursor.fetchone()[0]
        
        # Get today's revenue
        today = datetime.date.today().isoformat()
        cursor.execute('''
            SELECT SUM(parking_fee) FROM parking_sessions 
            WHERE DATE(exit_time) = ? AND payment_status = 'PAID'
        ''', (today,))
        
        today_revenue = cursor.fetchone()[0] or 0
        
        conn.close()
        
        return {
            "zones": zones_status,
            "active_sessions": active_sessions,
            "today_revenue": round(today_revenue, 2)
        }
    
    def get_vehicle_parking_history(self, plate_number, limit=10):
        """Get parking history for a specific vehicle"""
        conn = sqlite3.connect(str(PARKING_DB_PATH))
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT ps.session_id, ps.entry_time, ps.exit_time, ps.parking_fee, 
                   ps.payment_status, pz.zone_name, psp.spot_number
            FROM parking_sessions ps
            JOIN parking_zones pz ON ps.zone_id = pz.id
            LEFT JOIN parking_spots psp ON ps.spot_id = psp.id
            WHERE ps.plate_number = ?
            ORDER BY ps.entry_time DESC
            LIMIT ?
        ''', (plate_number.upper().replace(' ', ''), limit))
        
        history = cursor.fetchall()
        conn.close()
        return history
    
    def register_vehicle(self, plate_number, owner_name, owner_phone=None, owner_email=None, 
                        vehicle_type="CAR", pass_type="MONTHLY", pass_duration_days=30):
        """Register a vehicle for monthly pass or VIP access"""
        conn = sqlite3.connect(str(PARKING_DB_PATH))
        cursor = conn.cursor()
        
        expiry_date = (datetime.datetime.now() + datetime.timedelta(days=pass_duration_days)).isoformat()
        
        try:
            cursor.execute('''
                INSERT INTO registered_vehicles 
                (plate_number, owner_name, owner_phone, owner_email, vehicle_type, 
                 pass_type, pass_expiry, registered_date)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                plate_number.upper().replace(' ', ''),
                owner_name,
                owner_phone,
                owner_email,
                vehicle_type,
                pass_type,
                expiry_date,
                datetime.datetime.now().isoformat()
            ))
            conn.commit()
            conn.close()
            return True, "Vehicle registered successfully"
        except sqlite3.IntegrityError:
            conn.close()
            return False, "Vehicle already registered"
    
    def check_vehicle_registration(self, plate_number):
        """Check if vehicle has active registration/pass"""
        conn = sqlite3.connect(str(PARKING_DB_PATH))
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT owner_name, pass_type, pass_expiry, vehicle_type
            FROM registered_vehicles 
            WHERE plate_number = ? AND is_active = 1
        ''', (plate_number.upper().replace(' ', ''),))
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            expiry_date = datetime.datetime.fromisoformat(result[2])
            is_expired = expiry_date < datetime.datetime.now()
            
            return {
                "registered": True,
                "owner_name": result[0],
                "pass_type": result[1],
                "pass_expiry": result[2],
                "vehicle_type": result[3],
                "is_expired": is_expired
            }
        else:
            return {"registered": False}
    
    def generate_violation(self, plate_number, violation_type, location, fine_amount, image_path=None):
        """Generate parking violation"""
        conn = sqlite3.connect(str(PARKING_DB_PATH))
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO parking_violations 
            (plate_number, violation_type, violation_time, location, fine_amount, image_path)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            plate_number.upper().replace(' ', ''),
            violation_type,
            datetime.datetime.now().isoformat(),
            location,
            fine_amount,
            image_path
        ))
        
        conn.commit()
        conn.close()
        return True

# Common violation types and fines
VIOLATION_TYPES = {
    'OVERSTAY': {'name': 'Parking Overstay', 'fine': 25.0},
    'NO_PAYMENT': {'name': 'No Payment Made', 'fine': 50.0},
    'WRONG_ZONE': {'name': 'Parked in Wrong Zone', 'fine': 30.0},
    'DISABLED_SPOT': {'name': 'Illegal Use of Disabled Spot', 'fine': 100.0},
    'BLOCKING': {'name': 'Blocking Traffic/Exit', 'fine': 75.0}
}

# Vehicle types and rates
VEHICLE_TYPES = {
    'CAR': {'name': 'Car', 'rate_multiplier': 1.0},
    'SUV': {'name': 'SUV', 'rate_multiplier': 1.2},
    'MOTORCYCLE': {'name': 'Motorcycle', 'rate_multiplier': 0.5},
    'TRUCK': {'name': 'Truck', 'rate_multiplier': 2.0},
    'ELECTRIC': {'name': 'Electric Vehicle', 'rate_multiplier': 0.8}
}