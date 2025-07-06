from peft.tuners.lora import Linear
import torch
import bitsandbytes as bnb  # Required if base_layer uses BnB Linear

class CustomLinear(Linear):
    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        previous_dtype = x.dtype
        model_status = kwargs.pop("model_status", "train")
        adapter_activation = kwargs.pop("adapter_activation", None)
        lora_hidden_states = kwargs.pop("lora_hidden_states", None)
        
        kwargs.pop("exit_layers", None)  # optionally passed but not used here

        hidden_states = x
        if self.disable_adapters and self.merged:
            self.unmerge()

        result = self.base_layer(hidden_states, *args, **kwargs)
        lora_result = result  # fallback, updated if adapters are used

        if not self.disable_adapters and not self.merged:
            for active_adapter in self.active_adapters:
                if active_adapter not in self.lora_A:
                    continue

                lora_A = self.lora_A[active_adapter]
                lora_B = self.lora_B[active_adapter]
                dropout = self.lora_dropout[active_adapter]
                scaling = self.scaling[active_adapter]

                if model_status == "eval":
                    assert lora_hidden_states is not None, "Missing lora_hidden_states in eval mode"
                    lora_input = lora_hidden_states.to(lora_A.weight.dtype)
                else:
                    lora_input = hidden_states.to(lora_A.weight.dtype)

                lora_output = lora_B(lora_A(dropout(lora_input))) * scaling

                if adapter_activation is not None:
                    lora_output *= adapter_activation.to(lora_output.device)

                if model_status == "eval":
                    if adapter_activation is not None and adapter_activation.item() == 1:
                        lora_result = result + lora_output
                    else:
                        lora_result = result + lora_output  # optionally replace with just lora_output
                else:
                    lora_result = result + lora_output

        lora_result = lora_result.to(previous_dtype)

        if model_status == "eval":
            return result, lora_result
        else:  # train
            return lora_result

# Patch the PEFT Linear module
Linear.forward = CustomLinear.forward
