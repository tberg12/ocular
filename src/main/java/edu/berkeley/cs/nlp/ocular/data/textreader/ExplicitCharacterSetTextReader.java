package edu.berkeley.cs.nlp.ocular.data.textreader;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * @author Dan Garrette (dhg@cs.utexas.edu)
 */
public class ExplicitCharacterSetTextReader implements TextReader {

//	public static final Set<String> PUNC = CollectionHelper.makeSet(Charset.SPACE, ";", ":", "\"", "'", "!", "?", "(", ")", "«", "»", "¡", "¿");
//	public static final Set<String> DIGITS = CollectionHelper.makeSet("0", "1", "2", "3", "4", "5", "6", "7", "8", "9");
//	public static final Set<String> ALPHABET = CollectionHelper.makeSet("A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "Ñ", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "ñ", "o", "p", "q", "r", "s", Charset.LONG_S, "t", "u", "v", "w", "x", "y", "z");
//	public static final Set<String> LIGATURES = CollectionHelper.makeSet("Æ", "æ", "Œ", "œ");
//	public static final Set<String> ALL_ALLOWED = CollectionHelper.setUnion(Charset.UNIV_PUNC, PUNC, DIGITS, ALPHABET, LIGATURES);

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
			if (allValidCharacters.contains(c) || Charset.SPACE.equals(c) || Charset.UNIV_PUNC.contains(c)) {
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
		return "RemoveDiacriticsTextReader(" + delegate + ")";
	}

}
