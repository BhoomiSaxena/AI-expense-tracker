from sqlalchemy import create_engine, Column, Integer, String, Float, Date
from sqlalchemy.orm import declarative_base, sessionmaker
import pandas as pd

Base = declarative_base()


class Transaction(Base):
    __tablename__ = "transactions"
    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(Date, nullable=False)
    description = Column(String, nullable=False)
    amount = Column(Float, nullable=False)
    category = Column(String, nullable=False)


def init_db(db_url: str = "sqlite:///expense_tracker.db"):
    """Initialize database and return a sessionmaker."""
    engine = create_engine(db_url, connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)


def add_transactions(session_maker, df: pd.DataFrame):
    """Bulk-insert transactions from a DataFrame into the DB.

    Expects DataFrame columns: Date, Description, Amount, Category
    """
    Session = session_maker()
    try:
        mappings = []
        for _, r in df.iterrows():
            date_val = r.get("Date")
            if isinstance(date_val, pd.Timestamp):
                date_val = date_val.date()
            else:
                date_val = pd.to_datetime(date_val, dayfirst=True, errors="coerce").date()

            mappings.append(
                {
                    "date": date_val,
                    "description": r.get("Description"),
                    "amount": float(r.get("Amount")),
                    "category": r.get("Category"),
                }
            )

        if mappings:
            with Session.begin():
                Session.bulk_insert_mappings(Transaction, mappings)
    finally:
        Session.close()


def get_all_transactions_df(session_maker) -> pd.DataFrame:
    Session = session_maker()
    try:
        rows = Session.query(Transaction).order_by(Transaction.date).all()
        data = []
        for t in rows:
            data.append(
                {
                    "Date": pd.to_datetime(t.date),
                    "Description": t.description,
                    "Amount": t.amount,
                    "Category": t.category,
                }
            )
        return pd.DataFrame(data)
    finally:
        Session.close()
