package edu.berkeley.cs.nlp.ocular.data.textreader;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * @author Dan Garrette (dhgarrette@gmail.com)
 */
public class ExplicitCharacterSetTextReader implements TextReader {

	private Set<String> allValidCharacters;
	private TextReader delegate;

	public ExplicitCharacterSetTextReader(TextReader delegate, Set<String> validCharacters) {
		this.delegate = delegate;
		
		this.allValidCharacters = new HashSet<String>();
		this.allValidCharacters.addAll(validCharacters);
		this.allValidCharacters.add(Charset.SPACE);
		this.allValidCharacters.addAll(Charset.UNIV_PUNC);
	}

	public List<String> readCharacters(String line) {
		List<String> chars = new ArrayList<String>();
		for (String c : delegate.readCharacters(line)) {
  		/* Just check the base character, ignoring diacritics. If it is  
  		 * necessary to remove diacritics, the use RemoveDiacriticsTextReader
  		 * as well.
  		 */
			if (allValidCharacters.contains(c)) {
				chars.add(c);
			}
			else {
				String withoutDiacritic = Charset.removeAnyDiacriticFromChar(c);
				if (allValidCharacters.contains(withoutDiacritic)) {
					chars.add(withoutDiacritic);
				}
			}
		}
		return chars;
	}

	public String toString() {
		return "ExplicitCharacterSetTextReader(" + delegate + ")";
	}

}
