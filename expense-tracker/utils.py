import pandas as pd

INCOME_CATEGORIES = {"income", "salary", "freelance", "interest", "refund"}

def load_data(file):
    # Auto-detect separators (comma, tab, semicolon, etc.) from uploaded bank files.
    df = pd.read_csv(file, sep=None, engine='python')

    # Normalize column names for flexible matching.
    normalized_cols = {
        col: col.strip().lower().replace(" ", "").replace("_", "") for col in df.columns
    }

    aliases = {
        'Date': {'date', 'transactiondate', 'valuedate'},
        'Description': {'description', 'narration', 'details', 'remarks'},
        'Amount': {'amount', 'transactionamount', 'debitcreditamount', 'value'},
    }

    rename_map = {}
    for expected, possible_names in aliases.items():
        matched = next(
            (original for original, norm in normalized_cols.items() if norm in possible_names),
            None,
        )
        if matched is not None:
            rename_map[matched] = expected

    df = df.rename(columns=rename_map)

    missing = [col for col in ['Date', 'Description', 'Amount'] if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {', '.join(missing)}. Found: {list(df.columns)}")

    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
    df = df.dropna(subset=['Date'])
    return df

def clean_data(df):
    df = df.dropna()
    df['Amount'] = df['Amount'].astype(float)
    return df

def categorize(description):
    description = description.lower()

    if "swiggy" in description or "zomato" in description:
        return "Food"
    elif "uber" in description:
        return "Transport"
    elif "amazon" in description:
        return "Shopping"
    elif "salary" in description:
        return "Salary"
    elif "freelance" in description:
        return "Freelance"
    elif "interest" in description:
        return "Interest"
    elif "refund" in description:
        return "Refund"
    elif "bill" in description:
        return "Bills"
    else:
        return "Others"

def add_category(df):
    df['Category'] = df['Description'].apply(categorize)
    return df

def categorize(description):
    description = description.lower()

    if "swiggy" in description or "zomato" in description:
        return "Food"
    elif "uber" in description:
        return "Transport"
    elif "amazon" in description:
        return "Shopping"
    elif "salary" in description:
        return "Salary"
    elif "freelance" in description:
        return "Freelance"
    elif "interest" in description:
        return "Interest"
    elif "refund" in description:
        return "Refund"
    elif "bill" in description:
        return "Bills"
    else:
        return "Others"

def add_category(df):
    df['Category'] = df['Description'].apply(categorize)
    return df

def is_income_category(category_series: pd.Series) -> pd.Series:
    return category_series.astype(str).str.strip().str.lower().isin(INCOME_CATEGORIES)


def get_summary(df):
    income_mask = is_income_category(df['Category'])
    total_income = df.loc[income_mask, 'Amount'].abs().sum()
    total_expense = df.loc[~income_mask, 'Amount'].abs().sum()
    return total_income, total_expense

def category_summary(df):
    income_mask = is_income_category(df['Category'])
    expense_df = df.loc[~income_mask].copy()
    expense_df['Spend'] = expense_df['Amount'].abs()
    return expense_df.groupby('Category')['Spend'].sum()