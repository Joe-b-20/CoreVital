# ============================================================================
# CoreVital - Instrumentation Collector
#
# Purpose: Orchestrate model inference with instrumentation and data collection
# Inputs: Config, prompt text
# Outputs: InstrumentationResults with captured data
# Dependencies: torch, transformers, models, summaries, config
# Usage: collector = InstrumentationCollector(config); results = collector.run(prompt)
#
# Changelog:
#   2026-01-13: Initial collector for Phase-0
#   2026-01-14: Fixed logits extraction from generation outputs (added output_scores=True)
#                Fixed hidden_states and attentions extraction (handle tuple-of-tuples structure)
#                Added diagnostic logging for attention extraction
# ============================================================================

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import time
import torch

from CoreVital.config import Config
from CoreVital.models.hf_loader import load_model, ModelBundle
from CoreVital.instrumentation.summaries import (
    compute_hidden_summary,
    compute_attention_summary,
    compute_logits_summary,
)
from CoreVital.errors import InstrumentationError
from CoreVital.logging_utils import get_logger


logger = get_logger(__name__)


@dataclass
class StepData:
    """Data captured for a single generation step."""
    step_index: int
    token_id: int
    token_text: str
    is_prompt_token: bool
    logits: Optional[torch.Tensor] = None
    hidden_states: Optional[List[torch.Tensor]] = None
    attentions: Optional[List[torch.Tensor]] = None


@dataclass
class InstrumentationResults:
    """Complete results from an instrumented run."""
    model_bundle: ModelBundle
    prompt_text: str
    prompt_token_ids: List[int]
    generated_token_ids: List[int]
    generated_text: str
    timeline: List[StepData] = field(default_factory=list)
    elapsed_ms: int = 0
    warnings: List[Dict[str, str]] = field(default_factory=list)


class InstrumentationCollector:
    """
    Main collector that orchestrates instrumented inference.
    
    This class loads the model, runs generation with full instrumentation,
    and collects all necessary data for report generation.
    """
    
    def __init__(self, config: Config):
        """
        Initialize collector with configuration.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.model_bundle: Optional[ModelBundle] = None
    
    def run(self, prompt: str) -> InstrumentationResults:
        """
        Run instrumented inference on the given prompt.
        
        Args:
            prompt: Input prompt text
            
        Returns:
            InstrumentationResults with all collected data
            
        Raises:
            InstrumentationError: If inference fails
        """
        try:
            # Load model if not already loaded
            if self.model_bundle is None:
                self.model_bundle = load_model(self.config)
            
            # Set random seed for reproducibility
            if self.config.generation.seed is not None:
                torch.manual_seed(self.config.generation.seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(self.config.generation.seed)
            
            # Tokenize prompt
            logger.info("Tokenizing prompt...")
            inputs = self.model_bundle.tokenizer(
                prompt,
                return_tensors="pt",
                padding=False,
            ).to(self.model_bundle.device)
            
            prompt_token_ids = inputs.input_ids[0].tolist()
            logger.info(f"Prompt tokens: {len(prompt_token_ids)}")
            
            # Prepare generation config
            gen_config = {
                "max_new_tokens": self.config.generation.max_new_tokens,
                "do_sample": self.config.generation.do_sample,
                "temperature": self.config.generation.temperature,
                "top_k": self.config.generation.top_k,
                "top_p": self.config.generation.top_p,
                "output_hidden_states": True,
                "output_attentions": True,
                "output_scores": True,  # Enable logits extraction
                "return_dict_in_generate": True,
                "pad_token_id": self.model_bundle.tokenizer.pad_token_id,
            }
            
            # Run generation with instrumentation
            logger.info("Starting instrumented generation...")
            start_time = time.time()
            
            with torch.no_grad():
                outputs = self.model_bundle.model.generate(
                    **inputs,
                    **gen_config,
                )
            
            elapsed_ms = int((time.time() - start_time) * 1000)
            logger.info(f"Generation complete in {elapsed_ms}ms")
            
            # Extract generated tokens
            generated_ids = outputs.sequences[0].tolist()
            generated_token_ids = generated_ids[len(prompt_token_ids):]
            
            # Decode generated text
            generated_text = self.model_bundle.tokenizer.decode(
                generated_token_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            
            logger.info(f"Generated tokens: {len(generated_token_ids)}")
            #logger.info(f"Generated text: {generated_text}")
            
            # Process timeline
            timeline = self._process_timeline(
                outputs,
                prompt_token_ids,
                generated_token_ids,
            )
            
            # Collect warnings
            warnings = self._collect_warnings(outputs)
            
            return InstrumentationResults(
                model_bundle=self.model_bundle,
                prompt_text=prompt,
                prompt_token_ids=prompt_token_ids,
                generated_token_ids=generated_token_ids,
                generated_text=generated_text,
                timeline=timeline,
                elapsed_ms=elapsed_ms,
                warnings=warnings,
            )
            
        except Exception as e:
            logger.exception("Instrumentation failed")
            raise InstrumentationError(
                "Failed during instrumented inference",
                details=str(e)
            ) from e
    
    def _process_timeline(
        self,
        outputs: Any,
        prompt_token_ids: List[int],
        generated_token_ids: List[int],
    ) -> List[StepData]:
        """
        Process model outputs into timeline of steps.
        
        Args:
            outputs: Model generation outputs
            prompt_token_ids: Prompt token IDs
            generated_token_ids: Generated token IDs
            
        Returns:
            List of StepData objects
        """
        if self.model_bundle is None:
            raise InstrumentationError("Model bundle not initialized")
        
        timeline = []
        all_token_ids = prompt_token_ids + generated_token_ids
        
        # Note: Transformers generation outputs can be complex
        # Structure:
        # - scores: tuple of logits tensors, one per generation step
        # - hidden_states: tuple where each element is a tuple of hidden state tensors (one per layer)
        # - attentions: tuple where each element is a tuple of attention tensors (one per layer)
        
        has_scores = hasattr(outputs, 'scores') and outputs.scores is not None
        has_hidden = hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None
        has_attention = hasattr(outputs, 'attentions') and outputs.attentions is not None
        
        if not has_scores:
            logger.warning("Scores (logits) not available in outputs")
        if not has_hidden:
            logger.warning("Hidden states not available in outputs")
        if not has_attention:
            logger.warning("Attention weights not available in outputs")
        
        # Process each step
        # Note: For generation, we typically only get outputs for generated tokens
        # For prompt tokens, we need to run a separate forward pass if needed
        
        # For Phase-0 simplification: we'll focus on generated tokens
        # Full timeline (including prompt) can be added in future phases
        
        for step_idx, token_id in enumerate(generated_token_ids):
            token_text = self.model_bundle.tokenizer.decode([token_id])
            
            step_data = StepData(
                step_index=len(prompt_token_ids) + step_idx,
                token_id=token_id,
                token_text=token_text,
                is_prompt_token=False,
                logits=None,
                hidden_states=None,
                attentions=None,
            )
            
            # Extract logits (scores) if available
            try:
                if has_scores and len(outputs.scores) > step_idx:
                    # scores[step_idx] is the logits tensor for this generation step
                    step_data.logits = outputs.scores[step_idx]
            except (IndexError, AttributeError, TypeError) as e:
                logger.debug(f"Could not extract logits for step {step_idx}: {e}")
            
            # Extract hidden states if available
            # hidden_states is a tuple where each element corresponds to a generation step
            # Each element is itself a tuple of tensors (one per layer)
            try:
                if has_hidden and len(outputs.hidden_states) > step_idx:
                    # Convert tuple of tensors to list for easier handling
                    step_hidden = outputs.hidden_states[step_idx]
                    if isinstance(step_hidden, (tuple, list)):
                        step_data.hidden_states = list(step_hidden)
                    else:
                        step_data.hidden_states = [step_hidden]
            except (IndexError, AttributeError, TypeError) as e:
                logger.debug(f"Could not extract hidden states for step {step_idx}: {e}")
            
            # Extract attentions if available
            # attentions is a tuple where each element corresponds to a generation step
            # Each element is itself a tuple of tensors (one per layer)
            try:
                if has_attention and len(outputs.attentions) > step_idx:
                    # Convert tuple of tensors to list for easier handling
                    step_attn = outputs.attentions[step_idx]
                    if isinstance(step_attn, (tuple, list)):
                        step_data.attentions = list(step_attn)
                        if len(step_data.attentions) > 0:
                            logger.debug(f"Extracted {len(step_data.attentions)} attention tensors for step {step_idx}")
                    else:
                        step_data.attentions = [step_attn]
                        logger.debug(f"Extracted single attention tensor for step {step_idx}")
                elif has_attention:
                    logger.debug(f"Attention available but step_idx {step_idx} >= len {len(outputs.attentions)}")
                else:
                    logger.debug(f"No attention available for step {step_idx}")
            except (IndexError, AttributeError, TypeError) as e:
                logger.warning(f"Could not extract attentions for step {step_idx}: {e}")
                import traceback
                logger.debug(traceback.format_exc())
            
            timeline.append(step_data)
        
        logger.info(f"Processed {len(timeline)} timeline steps")
        return timeline
    
    def _collect_warnings(self, outputs: Any) -> List[Dict[str, str]]:
        """
        Collect warnings based on what data was available.
        
        Args:
            outputs: Model generation outputs
            
        Returns:
            List of warning dictionaries
        """
        warnings = []
        
        has_scores = hasattr(outputs, 'scores') and outputs.scores is not None
        has_hidden = hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None
        has_attention = hasattr(outputs, 'attentions') and outputs.attentions is not None
        
        if not has_scores:
            warnings.append({
                "code": "SCORES_NOT_AVAILABLE",
                "message": "Model did not return scores (logits); logits_summary omitted."
            })
        
        if not has_attention:
            warnings.append({
                "code": "ATTENTION_NOT_AVAILABLE",
                "message": "Model did not return attentions; attention_summary omitted."
            })
        
        if not has_hidden:
            warnings.append({
                "code": "HIDDEN_STATES_NOT_AVAILABLE",
                "message": "Model did not return hidden_states; hidden_summary omitted."
            })
        
        return warnings


# ============================================================================
# Test Harness
# ============================================================================

def _test_collector():
    """Test harness for instrumentation collector."""
    print("Testing InstrumentationCollector...")
    
    from CoreVital.config import Config
    
    config = Config()
    config.model.hf_id = "gpt2"
    config.device.requested = "cpu"
    config.generation.max_new_tokens = 5
    
    collector = InstrumentationCollector(config)
    results = collector.run("Hello")
    
    print("✓ Collected results:")
    print(f"  Prompt tokens: {len(results.prompt_token_ids)}")
    print(f"  Generated tokens: {len(results.generated_token_ids)}")
    print(f"  Timeline steps: {len(results.timeline)}")
    print(f"  Elapsed: {results.elapsed_ms}ms")
    print(f"  Warnings: {len(results.warnings)}")
    
    assert len(results.timeline) > 0
    print("✓ All collector tests passed!\n")


if __name__ == "__main__":
    _test_collector()