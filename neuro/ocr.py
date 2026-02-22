"""
DeepSeek OCR Integration
========================

Provides OCR capabilities using deepseek-ocr:latest via Ollama.
- Read text from images/screenshots
- Extract content from PDFs
- Analyze diagrams and charts
- Learn from visual content

Used by both tools and autonomous learning.
"""

import os
import base64
import json
import requests
from pathlib import Path
from typing import Optional, Dict, List, Any
from dataclasses import dataclass
from datetime import datetime


@dataclass
class OCRResult:
    """Result from OCR processing."""

    text: str
    source: str
    confidence: float = 0.0
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class DeepSeekOCR:
    """
    DeepSeek OCR for reading text from images.

    Uses deepseek-ocr:latest via Ollama.
    """

    def __init__(
        self,
        model: str = "deepseek-ocr:latest",
        base_url: str = "http://localhost:11434",
        timeout: int = 60,
    ):
        self.model = model
        self.base_url = base_url
        self.timeout = timeout
        self.history = []

    def _encode_image(self, image_path: str) -> Optional[str]:
        """Encode image to base64."""
        path = Path(image_path)
        if not path.exists():
            return None

        try:
            with open(path, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        except Exception:
            return None

    def _call_ollama(self, prompt: str, image_base64: str) -> Optional[str]:
        """Call Ollama with image."""
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "images": [image_base64],
                    "stream": False,
                    "options": {
                        "temperature": 0.1,  # Low temp for accuracy
                    },
                },
                timeout=self.timeout,
            )

            if response.status_code == 200:
                return response.json().get("response", "")
            return None
        except Exception as e:
            print(f"[OCR] Ollama error: {e}")
            return None

    def read_image(self, image_path: str, prompt: str = None) -> OCRResult:
        """
        Read text from an image.

        Args:
            image_path: Path to image file
            prompt: Optional custom prompt (default: extract all text)

        Returns:
            OCRResult with extracted text
        """
        image_b64 = self._encode_image(image_path)
        if not image_b64:
            return OCRResult(
                text="",
                source=image_path,
                confidence=0.0,
                metadata={"error": "Could not read image"},
            )

        if prompt is None:
            prompt = """Extract ALL text from this image.
Include:
- Main text content
- Headers and titles
- Labels and captions
- Code if present
- Any visible text

Return the text exactly as it appears, preserving structure where possible."""

        result = self._call_ollama(prompt, image_b64)

        if result:
            self.history.append(
                {"path": image_path, "time": datetime.now().isoformat(), "text_length": len(result)}
            )

            return OCRResult(
                text=result,
                source=image_path,
                confidence=0.9,
                metadata={"model": self.model, "prompt": prompt[:100]},
            )

        return OCRResult(
            text="", source=image_path, confidence=0.0, metadata={"error": "OCR failed"}
        )

    def analyze_image(self, image_path: str, question: str) -> OCRResult:
        """
        Analyze an image and answer a question about it.

        Args:
            image_path: Path to image
            question: Question to answer about the image

        Returns:
            OCRResult with answer
        """
        image_b64 = self._encode_image(image_path)
        if not image_b64:
            return OCRResult(text="", source=image_path, confidence=0.0)

        prompt = f"""Look at this image and answer the following question:

Question: {question}

Provide a detailed, accurate answer based on what you see in the image."""

        result = self._call_ollama(prompt, image_b64)

        return OCRResult(
            text=result or "",
            source=image_path,
            confidence=0.85 if result else 0.0,
            metadata={"question": question},
        )

    def extract_code(self, image_path: str) -> OCRResult:
        """Extract code from a screenshot."""
        image_b64 = self._encode_image(image_path)
        if not image_b64:
            return OCRResult(text="", source=image_path, confidence=0.0)

        prompt = """Extract the code from this image.
Return ONLY the code, properly formatted with correct indentation.
If there are line numbers, remove them.
Preserve the exact syntax and structure."""

        result = self._call_ollama(prompt, image_b64)

        return OCRResult(
            text=result or "",
            source=image_path,
            confidence=0.9 if result else 0.0,
            metadata={"type": "code"},
        )

    def extract_diagram(self, image_path: str) -> OCRResult:
        """Extract information from a diagram/chart."""
        image_b64 = self._encode_image(image_path)
        if not image_b64:
            return OCRResult(text="", source=image_path, confidence=0.0)

        prompt = """Analyze this diagram/chart and extract:
1. Type of diagram (flowchart, architecture, graph, etc.)
2. All labels and text
3. Relationships between components
4. Key insights or data points

Provide a structured description."""

        result = self._call_ollama(prompt, image_b64)

        return OCRResult(
            text=result or "",
            source=image_path,
            confidence=0.85 if result else 0.0,
            metadata={"type": "diagram"},
        )

    def extract_for_learning(self, image_path: str) -> Dict[str, Any]:
        """
        Extract content from image for autonomous learning.

        Returns structured data suitable for learning pipeline.
        """
        image_b64 = self._encode_image(image_path)
        if not image_b64:
            return {"error": "Could not read image"}

        prompt = """Analyze this image and extract knowledge in JSON format:

{
    "type": "document|code|diagram|screenshot|other",
    "title": "brief title",
    "summary": "one sentence summary",
    "content": "main text content",
    "facts": ["fact 1", "fact 2"],
    "topics": ["topic1", "topic2"],
    "qa_pairs": [{"q": "question", "a": "answer"}]
}

Return ONLY valid JSON."""

        result = self._call_ollama(prompt, image_b64)

        if result:
            # Try to parse JSON
            try:
                # Clean up response
                result = result.strip()
                if result.startswith("```"):
                    result = result.split("```")[1]
                    if result.startswith("json"):
                        result = result[4:]

                data = json.loads(result)
                data["source"] = image_path
                data["extracted_at"] = datetime.now().isoformat()
                return data
            except json.JSONDecodeError:
                # Return as plain text
                return {"type": "text", "content": result, "source": image_path}

        return {"error": "OCR extraction failed"}

    def batch_process(self, image_paths: List[str]) -> List[OCRResult]:
        """Process multiple images."""
        results = []
        for path in image_paths:
            result = self.read_image(path)
            results.append(result)
        return results

    def is_available(self) -> bool:
        """Check if DeepSeek OCR is available."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                return any(self.model in m.get("name", "") for m in models)
            return False
        except Exception:
            return False

    def get_stats(self) -> Dict:
        """Get OCR statistics."""
        return {
            "model": self.model,
            "images_processed": len(self.history),
            "available": self.is_available(),
        }


# Global instance
_ocr: Optional[DeepSeekOCR] = None


def get_ocr() -> DeepSeekOCR:
    """Get global OCR instance."""
    global _ocr
    if _ocr is None:
        _ocr = DeepSeekOCR()
    return _ocr


# Test
if __name__ == "__main__":
    print("=" * 50)
    print("DeepSeek OCR Test")
    print("=" * 50)

    ocr = DeepSeekOCR()

    print(f"\nModel: {ocr.model}")
    print(f"Available: {ocr.is_available()}")

    # Test with a sample image if provided
    import sys

    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        print(f"\nProcessing: {image_path}")

        result = ocr.read_image(image_path)
        print(f"\nExtracted text ({len(result.text)} chars):")
        print("-" * 40)
        print(result.text[:500])
        if len(result.text) > 500:
            print("...")
    else:
        print("\nUsage: python ocr.py <image_path>")

    print("\n" + "=" * 50)
