package edu.berkeley.cs.nlp.ocular.lm;

import java.util.List;

public class Transcription {
	List<Integer> text;
	
	public Transcription(List<Integer> guess) {
		this.text = guess;
	}
	
	public int getCharAt(int i) {
		return text.get(i);
	}
	
	public int getDocLength() {
		return text.size();
	}
}
