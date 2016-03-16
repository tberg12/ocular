package edu.berkeley.cs.nlp.ocular.data.textreader;

import static edu.berkeley.cs.nlp.ocular.util.CollectionHelper.makeMap;
import static edu.berkeley.cs.nlp.ocular.util.CollectionHelper.makeSet;
import static edu.berkeley.cs.nlp.ocular.util.CollectionHelper.setUnion;
import static edu.berkeley.cs.nlp.ocular.util.CollectionHelper.*;
import static edu.berkeley.cs.nlp.ocular.util.StringHelper.*;
import static edu.berkeley.cs.nlp.ocular.util.Tuple2.Tuple2;
import static edu.berkeley.cs.nlp.ocular.util.Tuple3.Tuple3;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import edu.berkeley.cs.nlp.ocular.util.StringHelper;
import edu.berkeley.cs.nlp.ocular.util.Tuple2;
import edu.berkeley.cs.nlp.ocular.util.Tuple3;
import indexer.Indexer;

/**
 * @author Dan Garrette (dhgarrette@gmail.com)
 */
public class Charset {

	public static final String SPACE = " ";
	public static final String HYPHEN = "-";
	public static final Set<String> LOWERCASE_LATIN_LETTERS = makeSet("a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z");
	public static final Set<String> LOWERCASE_VOWELS = makeSet("a", "e", "i", "o", "u");
	public static final Map<String,String> LIGATURES = makeMap(Tuple2("Æ","AE"), Tuple2("æ","ae"), Tuple2("Œ","OE"), Tuple2("œ","oe"));
	public static final String LONG_S = "\u017F"; // ſ
	public static final Set<String> BANNED_CHARS = makeSet("@", "$", "%");
	/**
	 * Punctuation symbols that should be made available for any language, 
	 * regardless of whether they are seen in the language model training 
	 * material.
	 */
	public static final Set<String> UNIV_PUNC = makeSet("&", ".", ",", "[", "]", HYPHEN, "*", "§", "¶");
	/**
	 * Punctuation is anything that is not alphabetic or a digit.
	 */
	public static boolean isPunctuation(char c) {
		return !Character.isWhitespace(c) && !Character.isAlphabetic(c) && !Character.isDigit(c);
	}
	public static boolean isPunctuationChar(String s) {
		for (char c: removeAnyDiacriticFromChar(s).toCharArray())
			if (!isPunctuation(c)) return false;
		return true;
	}
	
	public static final String GRAVE_COMBINING = "\u0300";
	public static final String ACUTE_COMBINING = "\u0301";
	public static final String CIRCUMFLEX_COMBINING = "\u0302";
	public static final String TILDE_COMBINING = "\u0303";
	public static final String MACRON_COMBINING = "\u0304"; // shorter overline
	public static final String BREVE_COMBINING = "\u0306";
	public static final String DIAERESIS_COMBINING = "\u0308"; // == umlaut
	public static final String CEDILLA_COMBINING = "\u0327";
	public static final String MACRON_BELOW_COMBINING = "\0331";
	public static final Set<String> COMBINING_CHARS = makeSet(GRAVE_COMBINING, ACUTE_COMBINING, CIRCUMFLEX_COMBINING, TILDE_COMBINING, MACRON_COMBINING, BREVE_COMBINING, DIAERESIS_COMBINING, CEDILLA_COMBINING, MACRON_BELOW_COMBINING);
	public static final String COMBINING_CHARS_RANGE_START = "\u0300";
	public static final String COMBINING_CHARS_RANGE_END = "\u036F";

	public static boolean isCombiningChar(String c) {
		return COMBINING_CHARS_RANGE_START.compareTo(c) <= 0 && c.compareTo(COMBINING_CHARS_RANGE_END) <= 0;
	}

	public static final String GRAVE_ESCAPE = "\\`";
	public static final String ACUTE_ESCAPE = "\\'";
	public static final String CIRCUMFLEX_ESCAPE = "\\^";
	public static final String TILDE_ESCAPE = "\\~";
	public static final String MACRON_ESCAPE = "\\-"; // shorter overline
	public static final String BREVE_ESCAPE = "\\v";
	public static final String DIAERESIS_ESCAPE = "\\\""; // == umlaut
	public static final String CEDILLA_ESCAPE = "\\c";
	public static final String MACRON_BELOW_ESCAPE = "\\_";
	public static final Set<String> ESCAPE_CHARS = makeSet(GRAVE_ESCAPE, ACUTE_ESCAPE, CIRCUMFLEX_ESCAPE, TILDE_ESCAPE, MACRON_ESCAPE, BREVE_ESCAPE, DIAERESIS_ESCAPE, CEDILLA_ESCAPE, MACRON_BELOW_ESCAPE);

	public static String combiningToEscape(String combiningChar) {
		if (GRAVE_COMBINING.equals(combiningChar))
			return GRAVE_ESCAPE;
		else if (ACUTE_COMBINING.equals(combiningChar))
			return ACUTE_ESCAPE;
		else if (CIRCUMFLEX_COMBINING.equals(combiningChar))
			return CIRCUMFLEX_ESCAPE;
		else if (TILDE_COMBINING.equals(combiningChar))
			return TILDE_ESCAPE;
		else if (MACRON_COMBINING.equals(combiningChar))
			return MACRON_ESCAPE;
		else if (BREVE_COMBINING.equals(combiningChar))
			return BREVE_ESCAPE;
		else if (DIAERESIS_COMBINING.equals(combiningChar))
			return DIAERESIS_ESCAPE;
		else if (CEDILLA_COMBINING.equals(combiningChar))
			return CEDILLA_ESCAPE;
		else if (MACRON_BELOW_COMBINING.equals(combiningChar))
			return MACRON_BELOW_ESCAPE;
		else
			throw new RuntimeException("Unrecognized combining char: [" + combiningChar + "] (" + StringHelper.toUnicode(combiningChar) + ")");
	}

	public static String escapeToCombining(String escSeq) {
		if (GRAVE_ESCAPE.equals(escSeq))
			return GRAVE_COMBINING;
		else if (ACUTE_ESCAPE.equals(escSeq))
			return ACUTE_COMBINING;
		else if (CIRCUMFLEX_ESCAPE.equals(escSeq))
			return CIRCUMFLEX_COMBINING;
		else if (TILDE_ESCAPE.equals(escSeq))
			return TILDE_COMBINING;
		else if (MACRON_ESCAPE.equals(escSeq))
			return MACRON_COMBINING;
		else if (BREVE_ESCAPE.equals(escSeq))
			return BREVE_COMBINING;
		else if (DIAERESIS_ESCAPE.equals(escSeq))
			return DIAERESIS_COMBINING;
		else if (CEDILLA_ESCAPE.equals(escSeq))
			return CEDILLA_COMBINING;
		else if (MACRON_BELOW_ESCAPE.equals(escSeq))
			return MACRON_BELOW_COMBINING;
		else
			throw new RuntimeException("Unrecognized escape sequence: [" + escSeq + "]");
	}

	public static final Map<String, String> PRECOMPOSED_TO_ESCAPED_MAP = new HashMap<String, String>();
	static {
		PRECOMPOSED_TO_ESCAPED_MAP.put("à", "\\`a"); // \`a
		PRECOMPOSED_TO_ESCAPED_MAP.put("á", "\\'a"); // \'a
		PRECOMPOSED_TO_ESCAPED_MAP.put("â", "\\^a"); // \^a
		PRECOMPOSED_TO_ESCAPED_MAP.put("ä", "\\\"a"); // \"a
		PRECOMPOSED_TO_ESCAPED_MAP.put("ã", "\\~a"); // \~a
		PRECOMPOSED_TO_ESCAPED_MAP.put("ā", "\\-a"); // \-a
		PRECOMPOSED_TO_ESCAPED_MAP.put("ă", "\\va"); // \va

		PRECOMPOSED_TO_ESCAPED_MAP.put("è", "\\`e"); // \`e
		PRECOMPOSED_TO_ESCAPED_MAP.put("é", "\\'e"); // \'e
		PRECOMPOSED_TO_ESCAPED_MAP.put("ê", "\\^e"); // \^e
		PRECOMPOSED_TO_ESCAPED_MAP.put("ë", "\\\"e"); // \"e
		PRECOMPOSED_TO_ESCAPED_MAP.put("ẽ", "\\~e"); // \~e
		PRECOMPOSED_TO_ESCAPED_MAP.put("ē", "\\-e"); // \-e
		PRECOMPOSED_TO_ESCAPED_MAP.put("ĕ", "\\ve"); // \ve

		PRECOMPOSED_TO_ESCAPED_MAP.put("ì", "\\`i"); // \`i
		PRECOMPOSED_TO_ESCAPED_MAP.put("í", "\\'i"); // \'i
		PRECOMPOSED_TO_ESCAPED_MAP.put("î", "\\^i"); // \^i
		PRECOMPOSED_TO_ESCAPED_MAP.put("ï", "\\\"i"); // \"i
		PRECOMPOSED_TO_ESCAPED_MAP.put("ĩ", "\\~i"); // \~i
		PRECOMPOSED_TO_ESCAPED_MAP.put("ī", "\\-i"); // \-i
		PRECOMPOSED_TO_ESCAPED_MAP.put("ı", "\\ii"); // \ii
		PRECOMPOSED_TO_ESCAPED_MAP.put("ī", "\\-i"); // \-i
		PRECOMPOSED_TO_ESCAPED_MAP.put("ĭ", "\\vi"); // \vi

		PRECOMPOSED_TO_ESCAPED_MAP.put("ò", "\\`o"); // \`o
		PRECOMPOSED_TO_ESCAPED_MAP.put("ó", "\\'o"); // \'o
		PRECOMPOSED_TO_ESCAPED_MAP.put("ô", "\\^o"); // \^o
		PRECOMPOSED_TO_ESCAPED_MAP.put("ö", "\\\"o"); // \"o
		PRECOMPOSED_TO_ESCAPED_MAP.put("õ", "\\~o"); // \~o
		PRECOMPOSED_TO_ESCAPED_MAP.put("ō", "\\-o"); // \-o
		PRECOMPOSED_TO_ESCAPED_MAP.put("ŏ", "\\vo"); // \vo

		PRECOMPOSED_TO_ESCAPED_MAP.put("ù", "\\`u"); // \`u
		PRECOMPOSED_TO_ESCAPED_MAP.put("ú", "\\'u"); // \'u
		PRECOMPOSED_TO_ESCAPED_MAP.put("û", "\\^u"); // \^u
		PRECOMPOSED_TO_ESCAPED_MAP.put("ü", "\\\"u"); // \"u
		PRECOMPOSED_TO_ESCAPED_MAP.put("ũ", "\\~u"); // \~u
		PRECOMPOSED_TO_ESCAPED_MAP.put("ū", "\\-u"); // \-u
		PRECOMPOSED_TO_ESCAPED_MAP.put("ŭ", "\\vu"); // \vu

		PRECOMPOSED_TO_ESCAPED_MAP.put("ñ", "\\~n"); // \~n
		PRECOMPOSED_TO_ESCAPED_MAP.put("ç", "\\cc"); // \cc

		PRECOMPOSED_TO_ESCAPED_MAP.put("À", "\\`A"); // \`A
		PRECOMPOSED_TO_ESCAPED_MAP.put("Á", "\\'A"); // \'A
		PRECOMPOSED_TO_ESCAPED_MAP.put("Â", "\\^A"); // \^A
		PRECOMPOSED_TO_ESCAPED_MAP.put("Ä", "\\\"A"); // \"A
		PRECOMPOSED_TO_ESCAPED_MAP.put("Ã", "\\~A"); // \~A
		PRECOMPOSED_TO_ESCAPED_MAP.put("Ā", "\\-A"); // \-A
		PRECOMPOSED_TO_ESCAPED_MAP.put("Ă", "\\vA"); // \vA

		PRECOMPOSED_TO_ESCAPED_MAP.put("È", "\\`E"); // \`E
		PRECOMPOSED_TO_ESCAPED_MAP.put("É", "\\'E"); // \'E
		PRECOMPOSED_TO_ESCAPED_MAP.put("Ê", "\\^E"); // \^E
		PRECOMPOSED_TO_ESCAPED_MAP.put("Ë", "\\\"E"); // \"E
		PRECOMPOSED_TO_ESCAPED_MAP.put("Ẽ", "\\~E"); // \~E
		PRECOMPOSED_TO_ESCAPED_MAP.put("Ē", "\\-E"); // \-E
		PRECOMPOSED_TO_ESCAPED_MAP.put("Ĕ", "\\vE"); // \ve

		PRECOMPOSED_TO_ESCAPED_MAP.put("Ì", "\\`I"); // \`I
		PRECOMPOSED_TO_ESCAPED_MAP.put("Í", "\\'I"); // \'I
		PRECOMPOSED_TO_ESCAPED_MAP.put("Î", "\\^I"); // \^I
		PRECOMPOSED_TO_ESCAPED_MAP.put("Ï", "\\\"I"); // \"I
		PRECOMPOSED_TO_ESCAPED_MAP.put("Ĩ", "\\~I"); // \~I
		PRECOMPOSED_TO_ESCAPED_MAP.put("Ī", "\\-I"); // \-I
		PRECOMPOSED_TO_ESCAPED_MAP.put("Ĭ", "\\vI"); // \vI

		PRECOMPOSED_TO_ESCAPED_MAP.put("Ò", "\\`O"); // \`O
		PRECOMPOSED_TO_ESCAPED_MAP.put("Ó", "\\'O"); // \'O
		PRECOMPOSED_TO_ESCAPED_MAP.put("Ô", "\\^O"); // \^O
		PRECOMPOSED_TO_ESCAPED_MAP.put("Ö", "\\\"O"); // \"O
		PRECOMPOSED_TO_ESCAPED_MAP.put("Õ", "\\~O"); // \~O
		PRECOMPOSED_TO_ESCAPED_MAP.put("Ō", "\\-O"); // \-O
		PRECOMPOSED_TO_ESCAPED_MAP.put("Ŏ", "\\vO"); // \vO

		PRECOMPOSED_TO_ESCAPED_MAP.put("Ù", "\\`U"); // \`U
		PRECOMPOSED_TO_ESCAPED_MAP.put("Ú", "\\'U"); // \'U
		PRECOMPOSED_TO_ESCAPED_MAP.put("Û", "\\^U"); // \^U
		PRECOMPOSED_TO_ESCAPED_MAP.put("Ü", "\\\"U"); // \"U
		PRECOMPOSED_TO_ESCAPED_MAP.put("Ũ", "\\~U"); // \~U
		PRECOMPOSED_TO_ESCAPED_MAP.put("Ū", "\\-U"); // \-U
		PRECOMPOSED_TO_ESCAPED_MAP.put("Ŭ", "\\vU"); // \vU

		PRECOMPOSED_TO_ESCAPED_MAP.put("Ñ", "\\~N"); // \~N
		PRECOMPOSED_TO_ESCAPED_MAP.put("Ç", "\\cC"); // \cC

		// note: superscript is marked \s as in superscript o = \so and superscript r is \sr
		//note for "breve" (u over letter) mark \va
	}

	private static final Map<String, String> ESCAPED_TO_PRECOMPOSED_MAP = new HashMap<String, String>();
	static {
		for (Map.Entry<String, String> entry : PRECOMPOSED_TO_ESCAPED_MAP.entrySet())
			ESCAPED_TO_PRECOMPOSED_MAP.put(entry.getValue(), entry.getKey());
	}
	
	public static final Set<String> CHARS_THAT_CAN_BE_REPLACED = setUnion(LOWERCASE_LATIN_LETTERS, makeSet("ç")); // TODO: Change this?
	public static final Set<String> VALID_CHAR_SUBSTITUTIONS = LOWERCASE_LATIN_LETTERS; // TODO: Change this?
	public static final Set<String> CHARS_THAT_CAN_DOUBLED = LOWERCASE_LATIN_LETTERS; // TODO: Change this?
	public static final Set<String> CHARS_THAT_CAN_BE_DECORATED_WITH_AN_ELISION_TILDE = LOWERCASE_LATIN_LETTERS; // TODO: Change this?
	public static final Set<String> CHARS_THAT_CAN_BE_ELIDED = LOWERCASE_LATIN_LETTERS; // TODO: Change this?
	public static final Set<String> ESCAPE_DIACRITICS_THAT_CAN_BE_DISREGARDED = makeSet(GRAVE_ESCAPE, ACUTE_ESCAPE);
	public static final Set<String> LETTERS_WITH_DISREGARDEDABLE_DIACRITICS = LOWERCASE_VOWELS;
	
	public static Set<Integer> makePunctSet(Indexer<String> charIndexer) {
		Set<Integer> punctSet = new HashSet<Integer>();
		for (String c : charIndexer.getObjects()) {
			if (Charset.isPunctuationChar(c))
				punctSet.add(charIndexer.getIndex(c));
		}
		return punctSet;
	}
	public static Set<Integer> makeCanBeReplacedSet(Indexer<String> charIndexer) {
		Set<Integer> canBeReplaced = new HashSet<Integer>();
		for (String c : charIndexer.getObjects()) {
			if (Charset.CHARS_THAT_CAN_BE_REPLACED.contains(c))
				canBeReplaced.add(charIndexer.getIndex(c));
		}
		return canBeReplaced;
	}
	public static Set<Integer> makeValidSubstitutionCharsSet(Indexer<String> charIndexer) {
		Set<Integer> validSubstitutionChars = new HashSet<Integer>();
		for (String c : charIndexer.getObjects()) {
			if (Charset.VALID_CHAR_SUBSTITUTIONS.contains(c))
				validSubstitutionChars.add(charIndexer.getIndex(c));
		}
		return validSubstitutionChars;
	}
	public static Set<Integer> makeValidDoublableSet(Indexer<String> charIndexer) {
		Set<Integer> validDoublableChars = new HashSet<Integer>();
		for (String c : charIndexer.getObjects()) {
			if (Charset.CHARS_THAT_CAN_DOUBLED.contains(c))
				validDoublableChars.add(charIndexer.getIndex(c));
		}
		return validDoublableChars;
	}
	public static Set<Integer> makeCanBeElidedSet(Indexer<String> charIndexer) {
		Set<Integer> canBeElided = new HashSet<Integer>();
		for (String c : charIndexer.getObjects()) {
			if (Charset.CHARS_THAT_CAN_BE_ELIDED.contains(c))
				canBeElided.add(charIndexer.getIndex(c));
		}
		return canBeElided;
	}
	public static Map<Integer,Integer> makeAddTildeMap(Indexer<String> charIndexer) {
		Map<Integer,Integer> m = new HashMap<Integer, Integer>();
		for (String original : charIndexer.getObjects()) {
			Tuple2<List<String>,String> originalEscapedDiacriticsAndLetter = Charset.escapeCharSeparateDiacritics(original);
			String baseLetter = originalEscapedDiacriticsAndLetter._2;
			if (Charset.CHARS_THAT_CAN_BE_DECORATED_WITH_AN_ELISION_TILDE.contains(original)) {
					m.put(charIndexer.getIndex(original), charIndexer.getIndex(Charset.TILDE_ESCAPE + baseLetter));
			}
			else if (LETTERS_WITH_DISREGARDEDABLE_DIACRITICS.contains(baseLetter)) {
				for (String diacritic : originalEscapedDiacriticsAndLetter._1) {
					if (ESCAPE_DIACRITICS_THAT_CAN_BE_DISREGARDED.contains(diacritic)) {
						m.put(charIndexer.getIndex(original), charIndexer.getIndex(Charset.TILDE_ESCAPE + baseLetter));
						break;
					}
				}
			}
		}
		return m;
	}
	public static Map<Integer,List<Integer>> makeLigatureMap(Indexer<String> charIndexer) {
		Map<Integer,List<Integer>> m = new HashMap<Integer, List<Integer>>();
		for (Map.Entry<String,String> entry : Charset.LIGATURES.entrySet()) {
			List<String> ligature = readCharacters(entry.getKey());
			if (ligature.size() > 1) throw new RuntimeException("Ligature ["+entry.getKey()+"] has more than one character: "+ligature);
			List<Integer> l = new ArrayList<Integer>();
			for (String c : readCharacters(entry.getValue()))
				l.add(charIndexer.getIndex(c));
			m.put(charIndexer.getIndex(ligature.get(0)), l);
		}
		return m;
	}
	public static Map<Integer,Integer> makeDiacriticDisregardMap(Indexer<String> charIndexer) {
		Map<Integer,Integer> m = new HashMap<Integer,Integer>();
		for (String original : charIndexer.getObjects()) { // find accented letters
			Tuple2<List<String>,String> originalEscapedDiacriticsAndLetter = escapeCharSeparateDiacritics(original);
			String baseLetter = originalEscapedDiacriticsAndLetter._2;
			if (LETTERS_WITH_DISREGARDEDABLE_DIACRITICS.contains(baseLetter)) {
				for (String diacritic : originalEscapedDiacriticsAndLetter._1) {
					if (ESCAPE_DIACRITICS_THAT_CAN_BE_DISREGARDED.contains(diacritic)) {
						m.put(charIndexer.getIndex(original), charIndexer.getIndex(baseLetter));
						break;
					}
				}
			}
		}
		return m;
	}
	
	
	/**
	 * Get the character code including any escaped diacritics that precede 
	 * the letter and any unicode "combining characters" that follow it.
	 * 
	 * Precomposed accents are given the highest priority.  Combining characters 
	 * are interpreted as left-associative and high-priority, while escapes are 
	 * right-associative and low-priority.  So, for a letter x with precomposed
	 * diacritic 0, combining chars 1,2,3, and escapes 4,5,6, the input 654x123 
	 * becomes encoded (with escapes) as 6543210x, and decoded (with precomposed 
	 * and combining characters) as x01234656.
	 * 
	 * @param s	A single character, potentially with diacritics encoded in any 
	 * form (composed, precomposed, escaped).
	 * @return	A string representing a single fully-escaped character, with all 
	 * diacritics (combining and precomposed) converted to their equivalent escape 
	 * sequences.
	 * @throws RuntimeException if the parameter `s` does not represent a single
	 * (potentially composed or escaped) character.
	 */
	public static String escapeChar(String s) {
		Tuple2<List<String>, String> diacriticsAndLetter = escapeCharSeparateDiacritics(s);
		return StringHelper.join(diacriticsAndLetter._1) + diacriticsAndLetter._2;
	}

	/**
	 * @see edu.berkeley.cs.nlp.ocular.data.textreader.textreader.Charset.escapeChar
	 * 
	 * @param s	A single character, potentially with diacritics encoded in any 
	 * form (composed, precomposed, escaped).
	 * @return	A fully-escaped character, with all diacritics (combining and 
	 * precomposed) converted to their equivalent escape sequence and placed in
	 * a list to be returned with the bare letter.
	 * @throws RuntimeException if the parameter `s` does not represent a single
	 * (potentially composed or escaped) character.
	 */
	public static Tuple2<List<String>,String> escapeCharSeparateDiacritics(String s) {
		Tuple3<List<String>, String, Integer> letterAndLength = readDiacriticsAndLetterAt(s, 0);
		int length = letterAndLength._3;
		if (s.length() != length) throw new RuntimeException("Could not escape ["+s+"] because it contains more than one character ("+StringHelper.toUnicode(s)+")");
		return Tuple2(letterAndLength._1, letterAndLength._2);
	}

	/**
	 * Read a single character from the line, starting at the given offset.
	 * 
	 * @see edu.berkeley.cs.nlp.ocular.data.textreader.textreader.Charset.escapeChar
	 * 
	 * @param line	A line of text possibly containing characters with diacritics
	 * composed, precomposed, or escaped.
	 * @param offset	The offset point in `line` from which to start reading for a 
	 * character.
	 * @return	A fully-escaped character string, with all diacritics (combining
	 * and precomposed) converted to their equivalent escape sequences.  Also 
	 * return the length in the ORIGINAL string of the span used to produce this 
	 * escaped character (to use as an offset when scanning through the string).
	 */
	public static Tuple2<String, Integer> readCharAt(String line, int offset) {
		Tuple3<List<String>, String, Integer> result = readDiacriticsAndLetterAt(line, offset);
		String c = StringHelper.join(result._1) + result._2;
		int length = result._3;
		return Tuple2(c, length);
	}
	
	/**
	 * Read a single character from the line including a list of all its diacritics, 
	 * starting at the given offset.
	 * 
	 * @see edu.berkeley.cs.nlp.ocular.data.textreader.textreader.Charset.escapeChar
	 * 
	 * @param line	A line of text possibly containing characters with diacritics
	 * composed, precomposed, or escaped.
	 * @param offset	The offset point in `line` from which to start reading for a 
	 * character.
	 * @return	A fully-escaped character, with all diacritics (combining and 
	 * precomposed) converted to their equivalent escape sequence and put in a list,
	 * the base letter with all diacritics removed, and the length in the ORIGINAL 
	 * string of the span used to produce this escaped character (to use as an 
	 * offset when scanning through the string).
	 */
	public static Tuple3<List<String>, String, Integer> readDiacriticsAndLetterAt(String line, int offset) {
		int lineLen = line.length();
		if (offset >= lineLen) throw new RuntimeException("offset must be less than the line length");
		
		if (lineLen - offset >= 2 && line.substring(offset, offset + 2).equals("\\\\"))
			return Tuple3((List<String>)new ArrayList<String>(), "\\\\", 2); // "\\" is its own character (for "\"), not an escaped diacritic
		
		List<String> diacritics = new ArrayList<String>();

		// get any escape prefixes characters
		int i = offset;
		while (i < lineLen && line.charAt(i) == '\\') {
			if (i + 1 >= lineLen) throw new RuntimeException("expected more after escape symbol, but found nothing: " + i + "," + lineLen + " " + line.substring(Math.max(0, i - 10), i) + "[" + line.substring(i) + "]");
			String escape = line.substring(i, i + 2);
			diacritics.add(escape);
			i += 2; // accept the 2-character escape sequence
		}

		int combiningCharCodeInsertionPoint = diacritics.size();

		if (i >= lineLen) throw new RuntimeException("expected a letter after escape code, but found nothing: " + i + "," + lineLen + " " + line.substring(Math.max(0, i - 50), i) + "[" + line.substring(i) + "]");
		String letter = line.substring(i, i + 1);
		if (isCombiningChar(letter)) throw new RuntimeException("expected a letter, but found only a combining character: " + i + "," + lineLen + " " + line.substring(Math.max(0, i - 50), i) + "[" + line.substring(i) + "]");
		i += 1; // accept the letter itself

		// get any combining characters
		while (i < lineLen) {
			String next = line.substring(i, i + 1);
			if (!isCombiningChar(next)) break;
			String escape = combiningToEscape(next);
			diacritics.add(combiningCharCodeInsertionPoint, escape);
			i++; // accept the combining character
		}

		String deprecomposedChar = Charset.PRECOMPOSED_TO_ESCAPED_MAP.get(letter);
		if (deprecomposedChar == null) {
			return Tuple3(diacritics, letter, i - offset);
		}
		else {
			int dcLen = deprecomposedChar.length();
			int j = 0;
			while (j < dcLen && deprecomposedChar.charAt(j) == '\\') {
				if (j + 1 >= dcLen) throw new RuntimeException("de-precomposed character has no letter after its escape sequence: " + i + "," + lineLen + " " + line.substring(Math.max(0, i - 10), i) + "[" + line.substring(i) + "]   ----  ["+deprecomposedChar+"]");
				String escape = deprecomposedChar.substring(j, j + 2);
				diacritics.add(escape);
				j += 2; // accept the 2-character escape sequence
			}
			String letterOnly = deprecomposedChar.substring(j, j + 1);
			return Tuple3(diacritics, letterOnly, i - offset);
		}
	}

	/**
	 * Convert a string into a sequence of diacritic-escaped characters.
	 * 
	 * @see edu.berkeley.cs.nlp.ocular.data.textreader.textreader.Charset.escapeChar
	 * 
	 * @param line	A line of text possibly containing characters with diacritics
	 * composed, precomposed, or escaped.
	 * @return	A fully-escaped character string, with all diacritics (combining
	 * and precomposed) converted to their equivalent escape sequences.
	 */
	public static List<String> readCharacters(String line) {
		List<String> escapedChars = new ArrayList<String>();
		int i = 0;
		while (i < line.length()) {
			Tuple2<String, Integer> escapedCharAndLength = Charset.readCharAt(line, i);
			String c = escapedCharAndLength._1;
			int length = escapedCharAndLength._2;
			escapedChars.add(c);
			i += length; // advance to the next character
		}
		return escapedChars;
	}
	
	/**
	 * Convert diacritic escape sequences on a character into unicode precomposed and combining characters
	 */
	public static String unescapeChar(String c) {
		if (c.length() == 1) return c; // no escapes
		if (c.equals("\\\\")) return c;
		
		Tuple2<List<String>,String> escapedDiacriticsAndLetter = escapeCharSeparateDiacritics(c); // use escapes only (and make sure it's a valid character)
		List<String> diacritics = escapedDiacriticsAndLetter._1;
		String baseLetter = escapedDiacriticsAndLetter._2;
		
		StringBuilder b = new StringBuilder();
		
		// Attempt to make a precomposed letter, falling back to composed otherwise
		String lastDiacritic = diacritics.get(diacritics.size()-1);
		String precomposed = ESCAPED_TO_PRECOMPOSED_MAP.get(lastDiacritic + baseLetter); // last escape + letter
		if (precomposed != null)
			b.append(precomposed);
		else 
			b.append(baseLetter).append(escapeToCombining(lastDiacritic));

		// Handle the rest of the escaped diacritics
		for (int i = diacritics.size() - 2; i >= 0; i -= 1) {
			b.append(escapeToCombining(diacritics.get(i)));
		}
		
		return b.toString();
	}

	/**
	 * Convert diacritic escape sequences on a character into unicode precomposed and combining characters
	 */
	public static String unescapeCharPrecomposedOnly(String c) {
		if (c.length() == 1) return c; // no escapes
		if (c.equals("\\\\")) return c;
		
		Tuple2<List<String>,String> escapedDiacriticsAndLetter = escapeCharSeparateDiacritics(c); // use escapes only (and make sure it's a valid character)
		List<String> diacritics = escapedDiacriticsAndLetter._1;
		String baseLetter = escapedDiacriticsAndLetter._2;
		
		StringBuilder b = new StringBuilder(join(take(diacritics, diacritics.size()-1)));
		
		// Attempt to make a precomposed letter, falling back to composed otherwise
		String lastDiacritic = last(diacritics);
		String precomposed = ESCAPED_TO_PRECOMPOSED_MAP.get(lastDiacritic + baseLetter); // last escape + letter
		if (precomposed != null)
			b.append(precomposed);
		else 
			b.append(lastDiacritic).append(baseLetter);

		return b.toString();
	}

	public static String removeAnyDiacriticFromChar(String c) {
		return escapeCharSeparateDiacritics(c)._2;
	}

}
