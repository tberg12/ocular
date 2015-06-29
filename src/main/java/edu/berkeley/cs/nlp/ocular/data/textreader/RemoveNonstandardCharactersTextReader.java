package edu.berkeley.cs.nlp.ocular.data.textreader;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;

import edu.berkeley.cs.nlp.ocular.util.CollectionHelper;

/**
 * @author Dan Garrette (dhg@cs.utexas.edu)
 */
public class RemoveNonstandardCharactersTextReader implements TextReader {

	private TextReader delegate;
	
	public static final Set<String> PUNC = CollectionHelper.makeSet(Charset.SPACE, ";", ":", "\"", "'", "!", "?", "(", ")", "«", "»", "¡", "¿");
	public static final Set<String> DIGITS = CollectionHelper.makeSet("0", "1", "2", "3", "4", "5", "6", "7", "8", "9");
	public static final Set<String> ALPHABET = CollectionHelper.makeSet("A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", Charset.LONG_S, "t", "u", "v", "w", "x", "y", "z");
	public static final Set<String> LIGATURES = CollectionHelper.makeSet("Æ", "æ", "Œ", "œ");
	public static final Set<String> ALL_ALLOWED = CollectionHelper.setUnion(Charset.UNIV_PUNC, PUNC, DIGITS, ALPHABET, LIGATURES);

	public RemoveNonstandardCharactersTextReader(TextReader delegate) {
		this.delegate = delegate;
	}

	public List<String> readCharacters(String line) {
		List<String> chars = new ArrayList<String>();
		for (String c : delegate.readCharacters(line)) {
  		/* Just check the base character, ignoring diacritics. If it is  
  		 * necessary to remove diacritics, the use RemoveDiacriticsTextReader
  		 * as well.
  		 */
			String withoutDiacritic = Charset.removeAnyDiacriticFromChar(c);
			if (ALL_ALLOWED.contains(withoutDiacritic)) { 
				chars.add(c);
			}
		}
		return chars;
	}

	public String toString() {
		return "RemoveDiacriticsTextReader(" + delegate + ")";
	}

}
