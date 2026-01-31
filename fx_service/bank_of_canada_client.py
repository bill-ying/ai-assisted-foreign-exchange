"""
Bank of Canada API client for fetching exchange rates.
"""

import requests
from datetime import date
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class BankOfCanadaClient:
    """
    Client for the Bank of Canada Valet API.
    
    Provides methods to fetch exchange rate observations for USD/CAD pairs.
    API documentation: https://www.bankofcanada.ca/valet/docs
    """
    
    BASE_URL = "https://www.bankofcanada.ca/valet/observations"
    
    def __init__(self, timeout: int = 30):
        """
        Initialize the Bank of Canada API client.
        
        Args:
            timeout: Request timeout in seconds
        """
        self._timeout = timeout
        self._session = requests.Session()
    
    def get_observations(
        self,
        series_name: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None
    ) -> List[Dict[str, Any]]:
        """
        Fetch observations for a given series from Bank of Canada.
        
        Args:
            series_name: The series identifier (e.g., 'FXUSDCAD', 'FXCADUSD')
            start_date: Optional start date for the query range
            end_date: Optional end date for the query range
            
        Returns:
            List of observation dictionaries containing date and rate data
            
        Raises:
            requests.RequestException: If the API request fails
        """
        url = f"{self.BASE_URL}/{series_name}"
        params = {}
        
        if start_date:
            params['start_date'] = start_date.isoformat()
        if end_date:
            params['end_date'] = end_date.isoformat()
        
        logger.debug(f"Fetching from Bank of Canada: {url} with params {params}")
        
        response = self._session.get(url, params=params, timeout=self._timeout)
        response.raise_for_status()
        
        data = response.json()
        observations = data.get('observations', [])
        
        logger.debug(f"Received {len(observations)} observations")
        return observations
    
    def get_rate_for_date(
        self,
        series_name: str,
        lookup_date: date
    ) -> Optional[Dict[str, Any]]:
        """
        Get the exchange rate for a specific date.
        
        Args:
            series_name: The series identifier (e.g., 'FXUSDCAD')
            lookup_date: The date to look up
            
        Returns:
            Observation dictionary if found, None otherwise
        """
        observations = self.get_observations(
            series_name,
            start_date=lookup_date,
            end_date=lookup_date
        )
        
        date_str = lookup_date.isoformat()
        for obs in observations:
            if obs.get('d') == date_str:
                return obs
        
        return None
    
    def get_rates_for_year(
        self,
        series_name: str,
        year: int
    ) -> List[Dict[str, Any]]:
        """
        Get all exchange rates for a given year.
        
        Args:
            series_name: The series identifier
            year: The year to fetch rates for
            
        Returns:
            List of observation dictionaries for the year
        """
        start = date(year, 1, 1)
        end = date(year, 12, 31)
        return self.get_observations(series_name, start_date=start, end_date=end)
    
    def close(self):
        """Close the underlying session."""
        self._session.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
