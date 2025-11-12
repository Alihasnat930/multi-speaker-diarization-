"""
Clinical SOAP Note Generator using Local LLM.
Supports Mistral, Llama3, and other open-source models with LoRA fine-tuning.
"""
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
    BitsAndBytesConfig
)
from typing import List, Dict, Optional
import logging
import json
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SOAPNoteGenerator:
    """
    Generate structured SOAP (Subjective, Objective, Assessment, Plan) clinical notes
    from dental consultation transcripts using fine-tuned LLM.
    """
    
    def __init__(
        self,
        model_name: str = "mistralai/Mistral-7B-Instruct-v0.2",
        use_lora: bool = True,
        lora_path: Optional[str] = "models/dental_soap_lora",
        use_4bit: bool = True,
        device: str = "auto"
    ):
        """
        Initialize SOAP note generator.
        
        Args:
            model_name: Base model name (Mistral, Llama3, etc.)
            use_lora: Whether to load LoRA adapters
            lora_path: Path to fine-tuned LoRA weights for dental SOAP notes
            use_4bit: Use 4-bit quantization for memory efficiency
            device: Device to run on ('auto', 'cuda', 'cpu')
        """
        self.model_name = model_name
        self.use_lora = use_lora
        self.lora_path = lora_path
        
        logger.info(f"Loading model: {model_name}")
        
        # Configure quantization for efficiency
        if use_4bit and torch.cuda.is_available():
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )
        else:
            bnb_config = None
            
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Load model
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map=device,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                trust_remote_code=True
            )
            
            # Load LoRA adapters if specified
            if use_lora and lora_path and Path(lora_path).exists():
                logger.info(f"Loading LoRA adapters from {lora_path}")
                from peft import PeftModel
                self.model = PeftModel.from_pretrained(self.model, lora_path)
                logger.info("LoRA adapters loaded successfully")
            elif use_lora:
                logger.warning(f"LoRA path {lora_path} not found, using base model")
                
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            logger.info("Falling back to lighter model...")
            # Fallback to smaller model
            self.model = AutoModelForCausalLM.from_pretrained(
                "mistralai/Mistral-7B-Instruct-v0.2",
                device_map=device,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            
        # Create pipeline
        self.pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=1024,
            temperature=0.3,
            do_sample=True,
            top_p=0.95,
        )
        
        logger.info("SOAP generator initialized")
        
    def generate_soap_note(
        self,
        segments: List[Dict],
        patient_context: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Generate SOAP note from conversation segments.
        
        Args:
            segments: List of transcript segments with 'speaker', 'text' keys
            patient_context: Optional patient background info
            
        Returns:
            Dictionary with 'subjective', 'objective', 'assessment', 'plan' keys
        """
        # Format conversation
        conversation = self._format_conversation(segments)
        
        # Create prompt
        prompt = self._create_soap_prompt(conversation, patient_context)
        
        # Generate
        logger.info("Generating SOAP note...")
        response = self.pipeline(prompt)[0]['generated_text']
        
        # Extract SOAP sections
        soap_note = self._parse_soap_response(response)
        
        return soap_note
        
    def _format_conversation(self, segments: List[Dict]) -> str:
        """Format transcript segments into conversation string"""
        lines = []
        for seg in segments:
            speaker = seg.get('speaker_id', seg.get('speaker', 'Unknown'))
            text = seg.get('text', '')
            if text:
                lines.append(f"{speaker}: {text}")
        return "\n".join(lines)
        
    def _create_soap_prompt(
        self,
        conversation: str,
        patient_context: Optional[str] = None
    ) -> str:
        """
        Create prompt for SOAP note generation.
        Uses instruction format for Mistral/Llama models.
        """
        system_prompt = """You are an expert dental clinical documentation assistant. Your task is to extract a structured SOAP (Subjective, Objective, Assessment, Plan) clinical note from a dental consultation conversation.

Guidelines:
- Subjective: Patient's chief complaint, symptoms, history in their own words
- Objective: Clinical findings, measurements, observations by dentist
- Assessment: Diagnosis, clinical impression
- Plan: Treatment recommendations, next steps, follow-up

Be concise, accurate, and professional. Use proper dental terminology."""

        if patient_context:
            context_section = f"\n\nPatient Context:\n{patient_context}\n"
        else:
            context_section = ""
            
        user_prompt = f"""{context_section}
Conversation Transcript:
{conversation}

Based on the above dental consultation, generate a structured SOAP note:"""

        # Format for Mistral/Llama instruction format
        if "mistral" in self.model_name.lower() or "llama" in self.model_name.lower():
            full_prompt = f"<s>[INST] {system_prompt}\n\n{user_prompt} [/INST]"
        else:
            full_prompt = f"{system_prompt}\n\n{user_prompt}"
            
        return full_prompt
        
    def _parse_soap_response(self, response: str) -> Dict[str, str]:
        """
        Parse generated text into structured SOAP sections.
        """
        # Remove the prompt part if it's in the response
        if "[/INST]" in response:
            response = response.split("[/INST]")[-1].strip()
            
        soap_note = {
            'subjective': '',
            'objective': '',
            'assessment': '',
            'plan': ''
        }
        
        # Try to extract sections
        lines = response.split('\n')
        current_section = None
        
        for line in lines:
            line_lower = line.lower().strip()
            
            # Detect section headers
            if 'subjective' in line_lower and ':' in line:
                current_section = 'subjective'
                soap_note[current_section] = line.split(':', 1)[1].strip()
            elif 'objective' in line_lower and ':' in line:
                current_section = 'objective'
                soap_note[current_section] = line.split(':', 1)[1].strip()
            elif 'assessment' in line_lower and ':' in line:
                current_section = 'assessment'
                soap_note[current_section] = line.split(':', 1)[1].strip()
            elif 'plan' in line_lower and ':' in line:
                current_section = 'plan'
                soap_note[current_section] = line.split(':', 1)[1].strip()
            elif current_section and line.strip():
                # Continue current section
                soap_note[current_section] += ' ' + line.strip()
                
        # Clean up
        for key in soap_note:
            soap_note[key] = soap_note[key].strip()
            
        return soap_note
        
    def format_soap_note_text(self, soap_note: Dict[str, str]) -> str:
        """Format SOAP note as readable text"""
        return f"""
SOAP Clinical Note
==================

SUBJECTIVE:
{soap_note.get('subjective', 'N/A')}

OBJECTIVE:
{soap_note.get('objective', 'N/A')}

ASSESSMENT:
{soap_note.get('assessment', 'N/A')}

PLAN:
{soap_note.get('plan', 'N/A')}
        """.strip()


# Global instance for backward compatibility
_generator = None

def get_generator():
    """Get or create global SOAP generator instance"""
    global _generator
    if _generator is None:
        # Use lighter model for quick initialization
        # For production, use proper Mistral-7B or Llama3 with LoRA
        try:
            _generator = SOAPNoteGenerator(
                model_name="mistralai/Mistral-7B-Instruct-v0.2",
                use_lora=True,
                use_4bit=True
            )
        except Exception as e:
            logger.error(f"Failed to load full model: {e}")
            logger.info("Using fallback lightweight model")
            # Ultra-light fallback
            from transformers import pipeline
            _generator = pipeline('text2text-generation', model='google/flan-t5-base')
    return _generator


def summarize_conversation(segments: List[Dict]) -> str:
    """
    Legacy function for backward compatibility.
    Generates SOAP note from conversation segments.
    """
    try:
        generator = get_generator()
        if isinstance(generator, SOAPNoteGenerator):
            soap_note = generator.generate_soap_note(segments)
            return generator.format_soap_note_text(soap_note)
        else:
            # Fallback pipeline
            convo = "\n".join([f"[{s.get('speaker_id', 'S')}] {s.get('text','')}" for s in segments])
            prompt = f"Generate a clinical SOAP note from this dental consultation:\n\n{convo}\n\nSOAP:"
            return generator(prompt, max_length=512)[0]['generated_text']
    except Exception as e:
        logger.error(f"SOAP generation failed: {e}")
        return "Error generating SOAP note. Please review transcript manually."
