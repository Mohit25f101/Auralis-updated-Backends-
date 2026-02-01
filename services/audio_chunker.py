# ==============================
# 📄 services/audio_chunker.py
# ==============================
"""
Audio Chunking Service
Splits long audio files into manageable chunks for processing
Supports up to 5-minute (300 second) audio files
"""

import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class AudioChunk:
    """Represents a chunk of audio"""
    data: np.ndarray
    start_time: float
    end_time: float
    chunk_index: int
    is_first: bool
    is_last: bool


class AudioChunker:
    """
    Audio Chunking Service
    
    Features:
    - Splits long audio into overlapping chunks
    - Maintains context with overlap
    - Smart splitting at silence/pause points
    - Supports up to 5 minutes of audio
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        chunk_duration: float = 30.0,
        overlap_duration: float = 2.0,
        min_chunk_duration: float = 5.0
    ):
        """
        Initialize audio chunker
        
        Args:
            sample_rate: Audio sample rate
            chunk_duration: Target chunk duration in seconds
            overlap_duration: Overlap between chunks in seconds
            min_chunk_duration: Minimum chunk duration in seconds
        """
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.overlap_duration = overlap_duration
        self.min_chunk_duration = min_chunk_duration
        self.loaded = True
        
        print(f"✅ Audio Chunker loaded (chunk: {chunk_duration}s, overlap: {overlap_duration}s)")
    
    def load(self) -> bool:
        """Load the chunker"""
        return True
    
    def chunk(self, audio: np.ndarray) -> List[AudioChunk]:
        """
        Split audio into chunks
        
        Args:
            audio: Audio waveform as numpy array
            
        Returns:
            List of AudioChunk objects
        """
        total_samples = len(audio)
        total_duration = total_samples / self.sample_rate
        
        # If audio is short enough, return as single chunk
        if total_duration <= self.chunk_duration:
            return [AudioChunk(
                data=audio,
                start_time=0.0,
                end_time=total_duration,
                chunk_index=0,
                is_first=True,
                is_last=True
            )]
        
        chunks = []
        chunk_samples = int(self.chunk_duration * self.sample_rate)
        overlap_samples = int(self.overlap_duration * self.sample_rate)
        step_samples = chunk_samples - overlap_samples
        
        chunk_index = 0
        start_sample = 0
        
        while start_sample < total_samples:
            end_sample = min(start_sample + chunk_samples, total_samples)
            
            # Extract chunk
            chunk_data = audio[start_sample:end_sample]
            
            # Skip if chunk is too short (unless it's the last chunk)
            if len(chunk_data) < self.min_chunk_duration * self.sample_rate:
                if chunks:  # If we have previous chunks, extend the last one
                    break
            
            # Try to find a good split point (silence)
            if end_sample < total_samples:
                chunk_data, end_sample = self._find_split_point(
                    audio, start_sample, end_sample
                )
            
            chunks.append(AudioChunk(
                data=chunk_data,
                start_time=start_sample / self.sample_rate,
                end_time=end_sample / self.sample_rate,
                chunk_index=chunk_index,
                is_first=(chunk_index == 0),
                is_last=(end_sample >= total_samples)
            ))
            
            # Move to next chunk
            start_sample = end_sample - overlap_samples
            chunk_index += 1
            
            # Safety check to prevent infinite loop
            if chunk_index > 100:
                break
        
        # Mark last chunk
        if chunks:
            chunks[-1] = AudioChunk(
                data=chunks[-1].data,
                start_time=chunks[-1].start_time,
                end_time=chunks[-1].end_time,
                chunk_index=chunks[-1].chunk_index,
                is_first=chunks[-1].is_first,
                is_last=True
            )
        
        return chunks
    
    def _find_split_point(
        self,
        audio: np.ndarray,
        start_sample: int,
        end_sample: int,
        search_window: float = 0.5
    ) -> Tuple[np.ndarray, int]:
        """
        Find optimal split point (at silence/pause)
        
        Args:
            audio: Full audio array
            start_sample: Chunk start
            end_sample: Chunk end
            search_window: Window to search for silence (seconds)
            
        Returns:
            Tuple of (chunk_data, actual_end_sample)
        """
        search_samples = int(search_window * self.sample_rate)
        search_start = max(end_sample - search_samples, start_sample)
        
        # Calculate energy in small frames within search window
        frame_size = int(0.02 * self.sample_rate)  # 20ms frames
        min_energy = float('inf')
        best_split = end_sample
        
        for i in range(search_start, end_sample - frame_size, frame_size // 2):
            frame = audio[i:i + frame_size]
            energy = np.sqrt(np.mean(frame ** 2))
            
            if energy < min_energy:
                min_energy = energy
                best_split = i + frame_size // 2
        
        # Use best split point if it's significantly quieter
        threshold = 0.02
        if min_energy < threshold:
            end_sample = best_split
        
        return audio[start_sample:end_sample], end_sample
    
    def merge_results(
        self,
        chunk_results: List[Dict[str, Any]],
        chunks: List[AudioChunk]
    ) -> Dict[str, Any]:
        """
        Merge results from multiple chunks
        
        Args:
            chunk_results: List of analysis results for each chunk
            chunks: Original AudioChunk objects
            
        Returns:
            Merged analysis result
        """
        if not chunk_results:
            return {}
        
        if len(chunk_results) == 1:
            return chunk_results[0]
        
        # Merge transcriptions
        merged_text = self._merge_transcriptions(chunk_results, chunks)
        
        # Merge sounds (take union with max confidence)
        merged_sounds = self._merge_sounds(chunk_results)
        
        # Merge locations (vote based)
        merged_location = self._merge_locations(chunk_results)
        
        # Merge situations (priority based)
        merged_situation = self._merge_situations(chunk_results)
        
        # Merge emotions (weighted average)
        merged_emotions = self._merge_emotions(chunk_results, chunks)
        
        # Build merged result
        merged = {
            "text": merged_text,
            "sounds": merged_sounds,
            "location": merged_location["location"],
            "location_confidence": merged_location["confidence"],
            "situation": merged_situation["situation"],
            "situation_confidence": merged_situation["confidence"],
            "is_emergency": any(r.get("is_emergency", False) for r in chunk_results),
            "emotions": merged_emotions,
            "chunk_count": len(chunks),
            "total_duration": chunks[-1].end_time if chunks else 0
        }
        
        return merged
    
    def _merge_transcriptions(
        self,
        results: List[Dict],
        chunks: List[AudioChunk]
    ) -> str:
        """Merge transcriptions from chunks, handling overlap"""
        texts = []
        
        for i, result in enumerate(results):
            text = result.get("text", "")
            
            if not text:
                continue
            
            # For overlapping chunks, try to remove duplicate content
            if i > 0 and texts:
                text = self._remove_overlap(texts[-1], text)
            
            if text:
                texts.append(text)
        
        return " ".join(texts)
    
    def _remove_overlap(self, prev_text: str, curr_text: str) -> str:
        """Remove overlapping content between consecutive chunks"""
        if not prev_text or not curr_text:
            return curr_text
        
        prev_words = prev_text.split()
        curr_words = curr_text.split()
        
        # Look for overlap at end of prev and start of curr
        max_overlap = min(10, len(prev_words), len(curr_words))
        
        for overlap_size in range(max_overlap, 0, -1):
            prev_end = prev_words[-overlap_size:]
            curr_start = curr_words[:overlap_size]
            
            if prev_end == curr_start:
                # Found overlap, remove it from current
                return " ".join(curr_words[overlap_size:])
        
        return curr_text
    
    def _merge_sounds(self, results: List[Dict]) -> Dict[str, float]:
        """Merge sounds from all chunks"""
        all_sounds = {}
        
        for result in results:
            sounds = result.get("sounds", {})
            for sound, confidence in sounds.items():
                if sound not in all_sounds:
                    all_sounds[sound] = confidence
                else:
                    all_sounds[sound] = max(all_sounds[sound], confidence)
        
        # Sort by confidence
        return dict(sorted(all_sounds.items(), key=lambda x: x[1], reverse=True)[:15])
    
    def _merge_locations(self, results: List[Dict]) -> Dict[str, Any]:
        """Merge locations using voting"""
        location_votes = {}
        confidence_sum = {}
        
        for result in results:
            loc = result.get("location", "Unknown")
            conf = result.get("location_confidence", 0)
            
            if loc not in location_votes:
                location_votes[loc] = 0
                confidence_sum[loc] = 0
            
            location_votes[loc] += 1
            confidence_sum[loc] += conf
        
        # Get most voted location
        best_loc = max(location_votes, key=location_votes.get)
        avg_conf = confidence_sum[best_loc] / location_votes[best_loc]
        
        return {"location": best_loc, "confidence": avg_conf}
    
    def _merge_situations(self, results: List[Dict]) -> Dict[str, Any]:
        """Merge situations with priority for emergencies"""
        # Check for emergency first
        for result in results:
            if result.get("is_emergency", False):
                return {
                    "situation": result.get("situation", "Emergency"),
                    "confidence": result.get("situation_confidence", 0.9)
                }
        
        # Otherwise, vote
        situation_votes = {}
        confidence_sum = {}
        
        for result in results:
            sit = result.get("situation", "Unknown")
            conf = result.get("situation_confidence", 0)
            
            if sit not in situation_votes:
                situation_votes[sit] = 0
                confidence_sum[sit] = 0
            
            situation_votes[sit] += 1
            confidence_sum[sit] += conf
        
        best_sit = max(situation_votes, key=situation_votes.get)
        avg_conf = confidence_sum[best_sit] / situation_votes[best_sit]
        
        return {"situation": best_sit, "confidence": avg_conf}
    
    def _merge_emotions(
        self,
        results: List[Dict],
        chunks: List[AudioChunk]
    ) -> Dict[str, Any]:
        """Merge emotions with duration weighting"""
        total_duration = sum(c.end_time - c.start_time for c in chunks)
        
        merged_micro = {
            "urgency": 0, "excitement": 0, "hesitation": 0,
            "confidence": 0, "stress": 0, "calmness": 0
        }
        
        intensity_sum = 0
        
        for result, chunk in zip(results, chunks):
            duration = chunk.end_time - chunk.start_time
            weight = duration / total_duration
            
            emotions = result.get("emotions", {})
            micro = emotions.get("micro_emotions", {})
            
            for key in merged_micro:
                merged_micro[key] += micro.get(key, 0) * weight
            
            intensity_sum += emotions.get("intensity", 5) * weight
        
        # Get primary emotion (most common)
        emotion_counts = {}
        for result in results:
            emo = result.get("emotions", {}).get("primary_emotion", "neutral")
            emotion_counts[emo] = emotion_counts.get(emo, 0) + 1
        
        primary = max(emotion_counts, key=emotion_counts.get)
        
        return {
            "primary_emotion": primary,
            "micro_emotions": {k: round(v, 3) for k, v in merged_micro.items()},
            "intensity": round(intensity_sum)
        }
    
    def get_chunk_info(self, audio: np.ndarray) -> Dict[str, Any]:
        """Get information about how audio will be chunked"""
        total_duration = len(audio) / self.sample_rate
        
        if total_duration <= self.chunk_duration:
            return {
                "total_duration": total_duration,
                "chunk_count": 1,
                "needs_chunking": False,
                "chunk_duration": total_duration
            }
        
        # Calculate number of chunks
        effective_chunk = self.chunk_duration - self.overlap_duration
        chunk_count = int(np.ceil(total_duration / effective_chunk))
        
        return {
            "total_duration": total_duration,
            "chunk_count": chunk_count,
            "needs_chunking": True,
            "chunk_duration": self.chunk_duration,
            "overlap_duration": self.overlap_duration
        }