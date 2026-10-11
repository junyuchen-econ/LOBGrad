import os
import sys
import json
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import torch

class LOB_Dataset():
    """
    Loads order book data and converts it to centered feature vectors.
    """
    def __init__(
        self, 
        SYMBOL: str, 
        START_DATE: datetime, 
        END_DATE: datetime, 
        LEVELS: int, 
        PRICE_LEVEL: float, 
        TIME_DIFF: int, 
        DEPTH: int = 100
    ) -> None:
        self.SYMBOL = SYMBOL
        self.START_DATE = START_DATE
        self.END_DATE = END_DATE
        self.LEVELS = LEVELS
        self.PRICE_LEVEL = PRICE_LEVEL
        self.TIME_DIFF = TIME_DIFF
        self.DEPTH = DEPTH

        self.centered_quantities, self.mid_prices, self.times, self.spread = self.__get_data__()

    def __get_raw_data__(self):
        """
        Loads raw order book data from JSON files and samples at regular time intervals.
        """
        ALL_BIDS = []
        ALL_ASKS = []
        ALL_TIME = []
         
        #root_path = sys.path[0]
        current_date = self.START_DATE
        curr_time = 0
        
        while current_date <= self.END_DATE:
            # Format date as MM-DD for filename
            date_str = current_date.strftime('%m-%d')
            file_path = os.path.join('data', self.SYMBOL + f'_{date_str}.json') 
            
            # Read JSONL file (one JSON object per line)
            with open(file_path, 'r') as file:  
                for line in file:
                    data = json.loads(line)
                    timestamp = data.get('timestamp')

                    # Sample snapshot only if enough time has passed
                    if timestamp - curr_time > self.TIME_DIFF:
                        # Get bids and asks
                        bids = data.get('bids', [])
                        asks = data.get('asks',[])
                        # Keep only top 'depth' levels
                        ALL_BIDS.append(bids[:self.DEPTH])
                        ALL_ASKS.append(asks[:self.DEPTH])
                        ALL_TIME.append(timestamp)

                        # Update last sampled timestamp
                        curr_time = timestamp
            
            # Move to next day
            current_date += timedelta(days=1)
            
        # shape: (num_snapshots, depth, 2)
        return np.array(ALL_BIDS), np.array(ALL_ASKS), np.array(ALL_TIME)

    def center(self, bids: np.ndarray, asks: np.ndarray) -> tuple[np.ndarray, float]:
        """
        Centers order book data around the mid price with 0.1 price precision.

        Parameters:
        ----------
        bids: np.ndarray: (depth, 2)
        asks: np.ndarray: (depth, 2)
        
        Returns:
        -------
        centered: np.ndarray: (2*level)
        mid_price: float: (1)

        Example:
        --------
        If level=10 and price_level=0.1:
        - Creates array of size 20 (2*10)
        - Bins prices at intervals of 0.1 around mid price
        - Aggregates quantities falling into each bin
        """

        # Step 1: Calculate best bid/ask with 0.1 precision
        best_bid = np.floor(bids[0][0])
        best_ask = np.ceil(asks[0][0])
        mid_price = round((best_bid+best_ask)/2, 2)
        centered_mid_price = np.ceil(mid_price) # Round mid price up to nearest 0.1 for centering reference
        
        # Step 2: Initialize centered array (NO mid point)
        centered_quantities = np.zeros(2*self.LEVELS)

        # Step 3: Extract price columns and round to integers
        bid_prices = np.floor(bids[:,0])
        ask_prices = np.ceil(asks[:,0])

        # Step 4: Define price intervals for binning
        bid_intervals = centered_mid_price - np.arange(1, self.LEVELS+1) * self.PRICE_LEVEL # not including the mid price: [level-1, level-2, ..., 0]
        ask_intervals = centered_mid_price + np.arange(self.LEVELS) * self.PRICE_LEVEL # including the mid price: [level, level+1, ..., 2*level-1]
        
        # Step 5: Aggregate BID quantities using vectorized operations
        for i, interval in enumerate(bid_intervals):
            # For each bid interval, sum all quantities where bid_prices match the interval
            # Store as negative values (bids are below mid price)
            # Index mapping: level-1-i maps to [level-2, level-1, ..., 0]
            centered_quantities[self.LEVELS-1-i] = -np.sum(bids[:, 1][bid_prices==interval])

        # Step 6: Aggregate ASK quantities using vectorized operations
        # For each ask interval, sum all quantities where ask_prices match the interval
        # Store as positive values (asks are above mid price)
        # Index mapping: level+i maps to [level, level+1, ..., 2*level-1]
        for i, interval in enumerate(ask_intervals):
            centered_quantities[self.LEVELS+i] = np.sum(asks[:, 1][ask_prices==interval])
        
        return centered_quantities, mid_price

    def __get_data__(self):
        """
        Loads order book data and converts it to centered feature vectors.

        Returns:
        -------
        centered: np.ndarray: (2*level, num_snapshots)
        mid_price: np.ndarray: (num_snapshots)
        times: list: (num_snapshots)
        spread: np.ndarray: (num_snapshots)
        """

        CENTERED_QUANTITIES = []
        MID_PRICE = []
        TIMES = []
        SPREAD = []
        
        current_date = self.START_DATE
        curr_time = 0
        
        while current_date <= self.END_DATE:    
            date_str = current_date.strftime('%m-%d')
            file_path = os.path.join('data', self.SYMBOL + f'_{date_str}.json') 
            
            with open(file_path, 'r') as file:  
                for line in file:
                    data = json.loads(line)
                    timestamp = data.get('timestamp')

                    # Sample snapshot only if enough time has passed
                    if timestamp - curr_time > self.TIME_DIFF:
                        
                        # Get bids and asks
                        bids = data.get('bids', [])
                        asks = data.get('asks',[])
                        # Convert to numpy arrays and center around mid price
                        centered_quantities, mid_price = self.center(np.array(bids), np.array(asks))
                        
                        CENTERED_QUANTITIES.append(centered_quantities)
                        MID_PRICE.append(mid_price)
                        TIMES.append(timestamp)
                        SPREAD.append(asks[0][0] - bids[0][0])
                        
                        curr_time = timestamp
            
            current_date += timedelta(days=1)
            
        return np.array(CENTERED_QUANTITIES).T, np.array(MID_PRICE), np.array(TIMES), np.array(SPREAD)



