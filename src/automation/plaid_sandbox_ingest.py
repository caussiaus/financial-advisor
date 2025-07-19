"""
Plaid Sandbox Ingestion Script

Fetches sample bank statement data from Plaid's sandbox API and saves as CSV/JSON in /data/statements/real/.
"""

import os
import json
import logging
from datetime import datetime
from pathlib import Path
import pandas as pd
from plaid import Client
from plaid.model import Products, CountryCode
from plaid.api import plaid_api
from plaid.model.sandbox_public_token_create_request import SandboxPublicTokenCreateRequest
from plaid.model.item_public_token_exchange_request import ItemPublicTokenExchangeRequest
from plaid.model.transactions_get_request import TransactionsGetRequest
from plaid.model.transactions_get_request_options import TransactionsGetRequestOptions
from plaid.model.accounts_get_request import AccountsGetRequest
from plaid.model.accounts_get_request_options import AccountsGetRequestOptions
from plaid.model.transactions_get_response import TransactionsGetResponse
from plaid.model.accounts_get_response import AccountsGetResponse
from plaid.model.error import Error as PlaidError
import requests

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Config (replace with your Plaid sandbox credentials or use env vars)
PLAID_CLIENT_ID = os.getenv("PLAID_CLIENT_ID", "sandbox-client-id")
PLAID_SECRET = os.getenv("PLAID_SECRET", "sandbox-secret")
PLAID_ENV = os.getenv("PLAID_ENV", "sandbox")

# Output directory
OUTPUT_DIR = Path("data/statements/real/")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Plaid API setup
def get_plaid_client():
    configuration = plaid_api.Configuration(
        host=plaid_api.Environment.Sandbox,
        api_key={
            'clientId': PLAID_CLIENT_ID,
            'secret': PLAID_SECRET,
        }
    )
    api_client = plaid_api.ApiClient(configuration)
    return plaid_api.PlaidApi(api_client)


def fetch_sandbox_transactions(start_date: str, end_date: str, count: int = 100):
    client = get_plaid_client()
    # Step 1: Create a sandbox public token
    request = SandboxPublicTokenCreateRequest(
        institution_id="ins_109508",  # Plaid's sandbox institution
        initial_products=[Products("transactions")]
    )
    response = client.sandbox_public_token_create(request)
    public_token = response['public_token']
    logger.info(f"Obtained sandbox public token: {public_token}")

    # Step 2: Exchange for access token
    exchange_request = ItemPublicTokenExchangeRequest(public_token=public_token)
    exchange_response = client.item_public_token_exchange(exchange_request)
    access_token = exchange_response['access_token']
    logger.info(f"Obtained access token: {access_token}")

    # Step 3: Fetch transactions
    transactions_request = TransactionsGetRequest(
        access_token=access_token,
        start_date=start_date,
        end_date=end_date,
        options=TransactionsGetRequestOptions(count=count)
    )
    transactions_response = client.transactions_get(transactions_request)
    transactions = transactions_response['transactions']
    accounts = transactions_response['accounts']
    logger.info(f"Fetched {len(transactions)} transactions for {len(accounts)} accounts")

    # Save as CSV/JSON
    for account in accounts:
        account_id = account['account_id']
        account_name = account['name'].replace(' ', '_')
        account_transactions = [t for t in transactions if t['account_id'] == account_id]
        if not account_transactions:
            continue
        df = pd.DataFrame(account_transactions)
        csv_path = OUTPUT_DIR / f"plaid_{account_name}_{start_date}_{end_date}.csv"
        json_path = OUTPUT_DIR / f"plaid_{account_name}_{start_date}_{end_date}.json"
        df.to_csv(csv_path, index=False)
        df.to_json(json_path, orient='records', indent=2)
        logger.info(f"Saved {len(df)} transactions to {csv_path} and {json_path}")


def main():
    # Example: last 30 days
    end_date = datetime.now().date()
    start_date = (end_date - pd.Timedelta(days=30))
    fetch_sandbox_transactions(str(start_date), str(end_date))
    logger.info("Plaid sandbox ingestion complete.")


if __name__ == "__main__":
    main() 