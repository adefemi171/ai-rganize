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
                 llm_provider: str = 'openai', max_folders: Optional[int] = None):
        """Initialize AI organizer with rate limiting and cost controls."""
        super().__init__(max_file_size_mb)
        
        self.batch_size = batch_size
        self.max_cost = max_cost
        self.current_cost = 0.0
        self.max_folders = max_folders
        self.ai_client = create_ai_client(llm_provider, api_key, model)
    
    def create_organization_plan(self, files: List[Dict], ai_limit: int = 50, verbose: bool = False) -> Dict:
        """Create organization plan using AI categorization."""
        plan = {}
        ai_limit_reached_message_shown = False
        existing_folders = set()  # Track folder names across batches for max_folders limit
        
        # Intelligent batching: adjust batch size based on cost and rate limits
        dynamic_batch_size = self._calculate_optimal_batch_size(files, ai_limit, verbose)
        
        if self.max_folders and verbose:
            print(f"📁 Maximum folders constraint: {self.max_folders} folders")
        
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
                
                # Calculate remaining folder slots if max_folders is set
                remaining_slots = None
                if self.max_folders:
                    remaining_slots = self.max_folders - len(existing_folders)
                    if remaining_slots <= 0:
                        if verbose:
                            print(f"⚠️  Maximum folder limit ({self.max_folders}) reached. All subsequent files will be assigned to existing folders.")
                        # Force reuse of existing folders
                        remaining_slots = 1  # Allow at least 1 folder (must reuse existing)
                
                folder_names = self.ai_client.categorize_files(
                    batch, 
                    verbose=verbose,
                    max_folders=self.max_folders,
                    existing_folders=list(existing_folders),
                    remaining_folder_slots=remaining_slots
                )
                
                # Update cost tracking from rate limiter
                self.current_cost = self.ai_client.rate_limiter.usage_stats['cost_estimate']
                
                # Add files to plan and track folder names
                for file_info, folder_name in zip(batch, folder_names):
                    clean_folder_name = self._clean_folder_name(folder_name)
                    original_folder_name = clean_folder_name
                    
                    # If max_folders is set and we've reached the limit, merge into existing folder
                    if self.max_folders and len(existing_folders) >= self.max_folders and clean_folder_name not in existing_folders:
                        # Find the most similar existing folder or use the first one
                        # For now, use the first existing folder as a fallback
                        clean_folder_name = list(existing_folders)[0] if existing_folders else clean_folder_name
                        if verbose:
                            print(f"    📦 Merging '{original_folder_name}' into existing folder: {clean_folder_name}")
                    
                    if clean_folder_name not in plan:
                        plan[clean_folder_name] = []
                    
                    plan[clean_folder_name].append(file_info)
                    # Only add to existing_folders if it's a new folder (to track unique count)
                    if clean_folder_name not in existing_folders:
                        existing_folders.add(clean_folder_name)
                
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
        """Calculate optimal batch size based on cost, rate limits, file characteristics, and max_folders."""
        total_files = len(files)
        
        # Smart batch size calculation: if max_folders is set, auto-calculate based on files/folders
        # Otherwise use the configured batch_size
        if self.max_folders is not None:
            # When max_folders is set, calculate batch size to distribute files intelligently
            # Strategy: Divide files by max_folders to get roughly one batch per folder category
            calculated_batch_size = total_files // self.max_folders
            
            # Ensure reasonable bounds: between 10 and 50 files per batch
            # Too small (< 10): Not enough context for LLM to understand patterns
            # Too large (> 50): May exceed token limits and be harder to process
            calculated_batch_size = max(10, min(50, calculated_batch_size))
            
            # Use calculated size if it's different from default, otherwise respect explicit batch_size
            # Only auto-calculate if user didn't explicitly set a custom batch_size (we check if it's the default 5)
            if self.batch_size == 5:  # Default value means user didn't specify
                optimal_size = calculated_batch_size
                if verbose:
                    num_batches = (total_files + optimal_size - 1) // optimal_size
                    print(f"📦 Auto-calculated batch size: {optimal_size} (based on {total_files} files and {self.max_folders} max folders → ~{num_batches} batches)")
            else:
                optimal_size = self.batch_size  # User explicitly set batch_size, use it
                if verbose:
                    print(f"📦 Using explicit batch size: {optimal_size} (max_folders constraint will still apply)")
        else:
            # No max_folders set: use configured batch_size
            optimal_size = self.batch_size
        
        # Adjust based on cost limits
        if self.current_cost > self.max_cost * 0.8:  # If approaching cost limit
            optimal_size = max(1, optimal_size // 2)  # Reduce batch size
            if verbose:
                print(f"💰 Reducing batch size to {optimal_size} due to cost limits")
        
        # Adjust based on file count vs AI limit
        if total_files > ai_limit * 2:  # Many files, use smaller batches
            # Don't force to 1, but reduce proportionally
            optimal_size = min(optimal_size, max(1, ai_limit // 3))
            if verbose:
                print(f"📊 Using smaller batch size {optimal_size} for large file set")
        
        # Ensure we don't exceed AI limit
        optimal_size = min(optimal_size, ai_limit)
        
        return optimal_size
    
