"""
Document analysis utilities for PDF and Word document content extraction.
"""

import tempfile
from pathlib import Path
from typing import Optional
import PyPDF2
try:
    from docx import Document
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False


class DocumentAnalyzer:
    """Analyzes PDF and Word documents for content and categorization."""
    
    def __init__(self, max_file_size_bytes: int):
        self.max_file_size_bytes = max_file_size_bytes
    
    def analyze_pdf(self, file_path: Path) -> str:
        """Analyze PDF files using hybrid approach."""
        try:
            file_size = file_path.stat().st_size / (1024 * 1024)  # MB
            analysis = f"PDF file: {file_size:.1f}MB"
            
            # Hybrid approach: Use OpenAI API for complex/large PDFs, local for simple ones
            if self._should_use_openai_api(file_path, file_size):
                try:
                    openai_analysis = self._analyze_pdf_with_openai_api(file_path)
                    if openai_analysis:
                        analysis += f" | Content: {openai_analysis}"
                except Exception as e:
                    # Fallback to local if OpenAI API fails
                    try:
                        local_analysis = self._analyze_pdf_local(file_path)
                        analysis += f" | Content: {local_analysis}"
                    except Exception as local_e:
                        analysis += f" | Analysis failed: {str(local_e)[:30]}"
            else:
                # Use local extraction for simple PDFs
                try:
                    local_analysis = self._analyze_pdf_local(file_path)
                    analysis += f" | Content: {local_analysis}"
                except Exception as local_e:
                    analysis += f" | Analysis failed: {str(local_e)[:30]}"
            
            return analysis
        except Exception:
            return "PDF file (analysis unavailable)"
    
    def analyze_word_document(self, file_path: Path) -> str:
        """Analyze Word documents using hybrid approach."""
        try:
            file_size = file_path.stat().st_size / 1024  # KB
            analysis = f"Word document: {file_size:.1f}KB"
            
            # Hybrid approach: Use OpenAI API for complex/large Word docs, local for simple ones
            if self._should_use_openai_api(file_path, file_size / 1024):  # Convert KB to MB
                try:
                    openai_analysis = self._analyze_word_with_openai_api(file_path)
                    if openai_analysis:
                        analysis += f" | Content: {openai_analysis}"
                except Exception as e:
                    # Fallback to local if OpenAI API fails
                    try:
                        local_analysis = self._analyze_word_local(file_path)
                        analysis += f" | Content: {local_analysis}"
                    except Exception as local_e:
                        analysis += f" | Analysis failed: {str(local_e)[:30]}"
            else:
                # Use local extraction for simple Word docs
                try:
                    local_analysis = self._analyze_word_local(file_path)
                    analysis += f" | Content: {local_analysis}"
                except Exception as local_e:
                    analysis += f" | Analysis failed: {str(local_e)[:30]}"
            
            return analysis
        except Exception:
            return "Word document (analysis unavailable)"
    
    def _should_use_openai_api(self, file_path: Path, file_size_mb: float) -> bool:
        """Determine if file should use OpenAI API based on complexity indicators."""
        # Use OpenAI API for:
        # 1. Large files (>2MB) that might need advanced processing
        # 2. Files that might be scanned (no text extraction with local methods)
        # 3. Complex documents (based on filename patterns)
        
        if file_size_mb > 2.0:  # Large files
            return True
        
        # Check for scanned document indicators in filename
        filename_lower = file_path.name.lower()
        scanned_indicators = ['scan', 'scanned', 'image', 'photo', 'picture']
        if any(indicator in filename_lower for indicator in scanned_indicators):
            return True
        
        # Check for complex document indicators
        complex_indicators = ['report', 'manual', 'handbook', 'guide', 'specification']
        if any(indicator in filename_lower for indicator in complex_indicators):
            return True
        
        # Default to local extraction for simple files
        return False
    
    def _analyze_pdf_with_openai_api(self, file_path: Path) -> str:
        """Analyze PDF using OpenAI File Search API."""
        try:
            from openai import OpenAI
            client = OpenAI()
            
            # Upload file to OpenAI
            with open(file_path, "rb") as file:
                uploaded_file = client.files.create(
                    file=file,
                    purpose="assistants"
                )
            
            # Create assistant with file search capability
            assistant = client.beta.assistants.create(
                model="gpt-4o",
                tools=[{"type": "file_search"}],
                tool_resources={
                    "file_search": {
                        "vector_store_ids": []
                    }
                }
            )
            
            # Create thread and analyze document
            thread = client.beta.threads.create()
            
            message = client.beta.threads.messages.create(
                thread_id=thread.id,
                role="user",
                content="Analyze this document and describe: 1) What type of document it is, 2) Key topics or themes, 3) The purpose or context, 4) Any important details. Keep it concise (max 150 words).",
                attachments=[
                    {
                        "file_id": uploaded_file.id,
                        "tools": [{"type": "file_search"}]
                    }
                ]
            )
            
            # Run the analysis
            run = client.beta.threads.runs.create(
                thread_id=thread.id,
                assistant_id=assistant.id
            )
            
            # Wait for completion
            import time
            while run.status in ["queued", "in_progress"]:
                time.sleep(1)
                run = client.beta.threads.runs.retrieve(
                    thread_id=thread.id,
                    run_id=run.id
                )
            
            if run.status == "completed":
                messages = client.beta.threads.messages.list(thread_id=thread.id)
                response = messages.data[0].content[0].text.value
                
                # Track document API costs
                file_size_mb = file_path.stat().st_size / (1024 * 1024)
                doc_cost = 0.10 + (file_size_mb * 0.01)  # Upload + processing cost
                # Note: Cost tracking would need to be passed from parent class
                
                # Cleanup
                client.files.delete(uploaded_file.id)
                client.beta.assistants.delete(assistant.id)
                
                return response
            else:
                return f"OpenAI analysis failed: {run.status}"
                
        except Exception as e:
            return f"OpenAI PDF analysis error: {str(e)[:50]}"
    
    def _analyze_pdf_local(self, file_path: Path) -> str:
        """Fallback local PDF analysis using PyPDF2."""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                # Extract text from first few pages
                text_content = ""
                max_pages = min(3, len(pdf_reader.pages))  # First 3 pages max
                
                for page_num in range(max_pages):
                    page = pdf_reader.pages[page_num]
                    text_content += page.extract_text() + " "
                
                # Clean and truncate text
                text_content = text_content.strip()[:500]  # First 500 chars
                
                if text_content:
                    # Look for common document patterns
                    if any(keyword in text_content.lower() for keyword in ['resume', 'cv', 'curriculum vitae']):
                        return f"PDF Resume/CV: {text_content[:100]}..."
                    elif any(keyword in text_content.lower() for keyword in ['cover letter', 'application']):
                        return f"PDF Cover Letter: {text_content[:100]}..."
                    elif any(keyword in text_content.lower() for keyword in ['invoice', 'receipt', 'bill']):
                        return f"PDF Financial Document: {text_content[:100]}..."
                    else:
                        return f"PDF Document: {text_content[:100]}..."
                else:
                    return "PDF Document (text extraction failed)"
            
        except Exception as e:
            return f"Local PDF analysis error: {str(e)[:50]}"
    
    def _analyze_word_with_openai_api(self, file_path: Path) -> str:
        """Analyze Word document using OpenAI File Search API."""
        try:
            from openai import OpenAI
            client = OpenAI()
            
            # Upload file to OpenAI
            with open(file_path, "rb") as file:
                uploaded_file = client.files.create(
                    file=file,
                    purpose="assistants"
                )
            
            # Create assistant with file search capability
            assistant = client.beta.assistants.create(
                model="gpt-4o",
                tools=[{"type": "file_search"}],
                tool_resources={
                    "file_search": {
                        "vector_store_ids": []
                    }
                }
            )
            
            # Create thread and analyze document
            thread = client.beta.threads.create()
            
            message = client.beta.threads.messages.create(
                thread_id=thread.id,
                role="user",
                content="Analyze this Word document and describe: 1) What type of document it is, 2) Key topics or themes, 3) The purpose or context, 4) Any important details. Keep it concise (max 150 words).",
                attachments=[
                    {
                        "file_id": uploaded_file.id,
                        "tools": [{"type": "file_search"}]
                    }
                ]
            )
            
            # Run the analysis
            run = client.beta.threads.runs.create(
                thread_id=thread.id,
                assistant_id=assistant.id
            )
            
            # Wait for completion
            import time
            while run.status in ["queued", "in_progress"]:
                time.sleep(1)
                run = client.beta.threads.runs.retrieve(
                    thread_id=thread.id,
                    run_id=run.id
                )
            
            if run.status == "completed":
                messages = client.beta.threads.messages.list(thread_id=thread.id)
                response = messages.data[0].content[0].text.value
                
                # Track document API costs
                file_size_mb = file_path.stat().st_size / (1024 * 1024)
                doc_cost = 0.10 + (file_size_mb * 0.01)  # Upload + processing cost
                # Note: Cost tracking would need to be passed from parent class
                
                # Cleanup
                client.files.delete(uploaded_file.id)
                client.beta.assistants.delete(assistant.id)
                
                return response
            else:
                return f"OpenAI analysis failed: {run.status}"
                
        except Exception as e:
            return f"OpenAI Word analysis error: {str(e)[:50]}"
    
    def _analyze_word_local(self, file_path: Path) -> str:
        """Fallback local Word document analysis using python-docx."""
        if not DOCX_AVAILABLE:
            return f"Word document: {file_path.stat().st_size / 1024:.1f}KB (python-docx not available)"
        
        try:
            doc = Document(file_path)
            
            # Extract text from all paragraphs
            text_content = ""
            for paragraph in doc.paragraphs:
                text_content += paragraph.text + " "
            
            # Clean and truncate text
            text_content = text_content.strip()[:500]  # First 500 chars
            
            if text_content:
                # Look for common document patterns
                if any(keyword in text_content.lower() for keyword in ['resume', 'cv', 'curriculum vitae']):
                    return f"Word Resume/CV: {text_content[:100]}..."
                elif any(keyword in text_content.lower() for keyword in ['cover letter', 'application']):
                    return f"Word Cover Letter: {text_content[:100]}..."
                elif any(keyword in text_content.lower() for keyword in ['invoice', 'receipt', 'bill']):
                    return f"Word Financial Document: {text_content[:100]}..."
                else:
                    return f"Word Document: {text_content[:100]}..."
            else:
                return "Word Document (text extraction failed)"
        
        except Exception as e:
            return f"Local Word document analysis error: {str(e)[:50]}"
