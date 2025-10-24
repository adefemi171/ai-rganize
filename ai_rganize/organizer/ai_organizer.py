"""
AI-powered file organizer using LLM for intelligent categorization.
"""

import time
from typing import Dict, List, Optional
from pathlib import Path

from .base_organizer import BaseOrganizer
from ..ai_client import create_ai_client, BaseAIClient


class AIOrganizer(BaseOrganizer):
    """AI-powered file organization using LLM for intelligent categorization."""
    
    def __init__(self, api_key: Optional[str] = None, max_file_size_mb: int = 10, 
                 batch_size: int = 5, max_cost: float = 1.0, model: str = 'gpt-4o', 
                 llm_provider: str = 'openai'):
        """Initialize AI organizer with rate limiting and cost controls."""
        super().__init__(max_file_size_mb)
        
        self.batch_size = batch_size
        self.max_cost = max_cost
        self.current_cost = 0.0
        self.ai_client = create_ai_client(llm_provider, api_key, model)
    
    def create_organization_plan(self, files: List[Dict], ai_limit: int = 50, verbose: bool = False) -> Dict:
        """Create organization plan using AI categorization."""
        plan = {}
        ai_limit_reached_message_shown = False
        
        # Intelligent batching: adjust batch size based on cost and rate limits
        dynamic_batch_size = self._calculate_optimal_batch_size(files, ai_limit, verbose)
        
        # Process files in batches with AI only
        for i in range(0, len(files), dynamic_batch_size):
            batch = files[i:i + dynamic_batch_size]
            
            # Check if we've reached the AI limit
            if i >= ai_limit and not ai_limit_reached_message_shown:
                if verbose:
                    print(f"🤖 AI limit reached ({ai_limit} files). Stopping AI processing.")
                ai_limit_reached_message_shown = True
                break  # Stop processing when AI limit is reached
            
            # Use AI for categorization
            try:
                if verbose:
                    print(f"🔄 Processing batch {i//self.batch_size + 1}/{(len(files) + self.batch_size - 1)//self.batch_size} ({len(batch)} files)")
                
                # Check cost limits and balance management
                if self.current_cost >= self.max_cost:
                    if verbose:
                        print(f"💰 Cost limit reached (${self.current_cost:.4f}), but continuing with AI processing")
                        print(f"⚠️  Balance management: Cost limit exceeded, consider reducing batch size or AI limit")
                
                # Balance management: adjust processing based on cost
                if self.current_cost > self.max_cost * 1.5:  # Significantly over limit
                    if verbose:
                        print(f"🚨 High cost detected (${self.current_cost:.4f}), using minimal batch size")
                    # Use single file batches to minimize cost
                    batch = batch[:1]
                
                folder_names = self.ai_client.categorize_files(batch, verbose=verbose)
                
                # Update cost tracking from rate limiter
                self.current_cost = self.ai_client.rate_limiter.usage_stats['cost_estimate']
                
                # Add files to plan
                for file_info, folder_name in zip(batch, folder_names):
                    clean_folder_name = self._clean_folder_name(folder_name)
                    
                    if clean_folder_name not in plan:
                        plan[clean_folder_name] = []
                    
                    plan[clean_folder_name].append(file_info)
                
                # Add small delay to respect rate limits
                time.sleep(0.1)
                
            except Exception as e:
                if verbose:
                    print(f"❌ AI categorization failed for batch: {e}")
                # Skip this batch - no fallback to rule-based
                if verbose:
                    print(f"⚠️  Skipping {len(batch)} files due to AI processing failure")
        
        # Add summary
        plan['summary'] = {
            'total_files': len(files),
            'total_folders': len([k for k in plan.keys() if k != 'summary']),
            'method': 'ai-powered',
            'ai_files_processed': min(ai_limit, len(files)),
            'cost_estimate': self.current_cost
        }
        
        return plan
    
    def _calculate_optimal_batch_size(self, files: List[Dict], ai_limit: int, verbose: bool) -> int:
        """Calculate optimal batch size based on cost, rate limits, and file characteristics."""
        # Start with configured batch size
        optimal_size = self.batch_size
        
        # Adjust based on cost limits
        if self.current_cost > self.max_cost * 0.8:  # If approaching cost limit
            optimal_size = max(1, optimal_size // 2)  # Reduce batch size
            if verbose:
                print(f"💰 Reducing batch size to {optimal_size} due to cost limits")
        
        # Adjust based on file count vs AI limit
        if len(files) > ai_limit * 2:  # Many files, use smaller batches
            # Don't force to 1, but reduce proportionally
            optimal_size = min(optimal_size, max(1, ai_limit // 3))
            if verbose:
                print(f"📊 Using smaller batch size {optimal_size} for large file set")
        
        # Ensure we don't exceed AI limit
        optimal_size = min(optimal_size, ai_limit)
        
        return optimal_size
    
