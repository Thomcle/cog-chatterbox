from cog import BasePredictor, Input, Path
from chatterbox.tts import ChatterboxTTS
import torchaudio

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        self.model = ChatterboxTTS.from_pretrained(device="cuda")

    # The arguments and types the model takes as input
    def predict(self,
        text:str,
        audio_prompt_path:Path = Input(),
        exaggeration:float = 0.5,
        cfg_weight:float = 0.5,
        temperature:float = 0.8
    ) -> Path:
        """Run a single prediction on the model"""

        wav = self.model.generate(
            text, 
            audio_prompt_path=audio_prompt_path,
            exaggeration=exaggeration,
            cfg_weight=cfg_weight,
            temperature=temperature
        )

        output_path = Path("/tmp/output.wav")
        torchaudio.save(output_path, wav, self.model.sr)

        return output_path