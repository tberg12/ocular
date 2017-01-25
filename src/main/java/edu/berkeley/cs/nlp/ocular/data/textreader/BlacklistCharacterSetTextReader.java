package edu.berkeley.cs.nlp.ocular.data.textreader;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * @author Dan Garrette (dhgarrette@gmail.com)
 */
public class BlacklistCharacterSetTextReader implements TextReader {

	private Set<String> allInvalidCharacters = new HashSet<String>();
	private TextReader delegate;

	public BlacklistCharacterSetTextReader(Set<String> invalidCharacters, TextReader delegate) {
		for (String c : invalidCharacters) {
			allInvalidCharacters.add(Charset.normalizeChar(c));
		}
		this.delegate = delegate;
	}

	public List<String> readCharacters(String line) {
		List<String> chars = new ArrayList<String>();
		for (String c : delegate.readCharacters(line)) {
			if (!allInvalidCharacters.contains(c)) {
				chars.add(c);
			}
		}
		return chars;
	}

	public String toString() {
		return "BlacklistCharacterSetTextReader(" + delegate + ")";
	}

}
