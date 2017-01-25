package edu.berkeley.cs.nlp.ocular.data.textreader;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * @author Dan Garrette (dhgarrette@gmail.com)
 */
public class WhitelistCharacterSetTextReader implements TextReader {

	private Set<String> allValidCharacters = new HashSet<String>();
	private boolean disregardDiacritics;
	private TextReader delegate;

	/**
	 * @param validCharacters	The set of characters that are allowed. 
	 * Any other character will be skipped.
	 * @param disregardDiacritics	If true, then a character with a diacritic
	 * will be considered valid even if only its non-diacritic version is in
	 * the validCharcters set.
	 * @param delegate
	 */
	public WhitelistCharacterSetTextReader(Set<String> validCharacters, boolean disregardDiacritics, TextReader delegate) {
		if (validCharacters.isEmpty()) {
			throw new RuntimeException("validCharacters is empty in WhitelistCharacterSetTextReader constructor");
		}
		
		for (String c : validCharacters) {
			allValidCharacters.add(Charset.normalizeChar(c));
		}
		allValidCharacters.add(Charset.SPACE);
		
		this.disregardDiacritics = disregardDiacritics;
		this.delegate = delegate;
	}

	public WhitelistCharacterSetTextReader(Set<String> validCharacters, TextReader delegate) {
		this(validCharacters, false, delegate);
	}

	public List<String> readCharacters(String line) {
		List<String> chars = new ArrayList<String>();
		for (String c : delegate.readCharacters(line)) {
			if (allValidCharacters.contains(c)) {
				chars.add(c);
			}
			else if (disregardDiacritics && allValidCharacters.contains(Charset.removeAnyDiacriticFromChar(c))) {
				chars.add(c);
			}
		}
		return chars;
	}

	public String toString() {
		return "WhitelistCharacterSetTextReader(" + delegate + ")";
	}

}
