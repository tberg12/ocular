package edu.berkeley.cs.nlp.ocular.data.textreader;

import java.util.ArrayList;
import java.util.List;

/**
 * @author Dan Garrette (dhgarrette@gmail.com)
 */
public class ConvertLongSTextReader implements TextReader {

	private TextReader delegate;

	public ConvertLongSTextReader(TextReader delegate) {
		this.delegate = delegate;
	}

	public List<String> readCharacters(String line) {
		List<String> chars = new ArrayList<String>();
		for (String c : delegate.readCharacters(line)) {
			chars.add(c);
		}

		/*
		 * Replace 's' characters with 'long-s' characters.
		 */
		// for every letter except the last (since the last letter can 
		//   never be a long-s since it can never be followed by a letter
		for (int t = 0; t < chars.size() - 1; t++) {
			if (chars.get(t).equals("s")) {
				String next = chars.get(t + 1);
				String nextWithoutDiacritics = Charset.removeAnyDiacriticFromChar(next);
				if (nextWithoutDiacritics.length() != 1) {
					if (!nextWithoutDiacritics.equals("\\\\")) {
						throw new AssertionError("expected nextWithoutDiacritics [" + nextWithoutDiacritics + "] length() == 1");
					}
				}
				char nextWithoutDiacriticsChar = nextWithoutDiacritics.charAt(0);
				if (t > 0 && chars.get(t - 1).equals(Charset.LONG_S) && nextWithoutDiacriticsChar == 'i') {
					// "ſsi": do nothing
				}
				else if (Character.isAlphabetic(nextWithoutDiacriticsChar)) {
					chars.set(t, Charset.LONG_S);
				}
			}
		}

		return chars;
	}
	
	public String toString() {
		return "ConvertLongSTextReader(" + delegate + ")";
	}

}
