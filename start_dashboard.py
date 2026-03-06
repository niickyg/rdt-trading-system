#!/usr/bin/env python3
"""
Start the RDT web dashboard in read-only mode.

Uses SQLite (same DB the bot writes to), skips broker/scanner init
so it doesn't conflict with the running bot's IBKR connection.

Usage: python start_dashboard.py [--port PORT]
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Load .env first to get SECRET_KEY etc.
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))

# Override DB to use SQLite instead of PostgreSQL
os.environ.pop('DATABASE_URL', None)
os.environ.pop('RDT_DATABASE_URL', None)
os.environ['RDT_ENV'] = 'development'
os.environ['FLASK_DEBUG'] = 'true'
os.environ.setdefault('WEBSOCKET_ASYNC_MODE', 'threading')
os.environ['RDT_SKIP_TRADING_INIT'] = 'true'

import argparse


def ensure_admin_user():
    """Create admin user in the auth database if it doesn't exist."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from data.database.models import Base, User
    from werkzeug.security import generate_password_hash
    from datetime import datetime

    # Auth DB lives at data/database/rdt_auth.db (matches web/auth.py get_db_path)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(base_dir, 'data', 'database', 'rdt_auth.db')
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    engine = create_engine(f'sqlite:///{db_path}', echo=False)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    try:
        existing = session.query(User).filter_by(username='admin').first()
        if not existing:
            admin = User(
                username='admin',
                email='admin@rdt.local',
                password_hash=generate_password_hash('RdtAdmin2026!@'),
                is_active=True,
                is_admin=True,
                created_at=datetime.utcnow(),
                failed_login_attempts=0,
                locked_until=None,
            )
            session.add(admin)
            session.commit()
            print("Created admin user: admin / RdtAdmin2026!@")
        else:
            print("Admin user already exists")
    finally:
        session.close()


def main():
    parser = argparse.ArgumentParser(description='Start RDT Dashboard')
    parser.add_argument('--port', type=int, default=5001, help='Port (default: 5001)')
    args = parser.parse_args()

    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex(('127.0.0.1', args.port))
    sock.close()
    if result == 0:
        print(f"Port {args.port} is already in use. Try --port {args.port + 1}")
        sys.exit(1)

    ensure_admin_user()

    print(f"\nStarting RDT Dashboard on http://localhost:{args.port}")
    print(f"Login: admin / RdtAdmin2026!@")
    print(f"Dashboard: http://localhost:{args.port}/dashboard/positions\n")

    os.environ['PORT'] = str(args.port)

    from web.app import app, socketio

    if socketio is not None:
        socketio.run(app, host='0.0.0.0', port=args.port, debug=False, allow_unsafe_werkzeug=True)
    else:
        app.run(host='0.0.0.0', port=args.port, debug=False)


if __name__ == '__main__':
    main()
