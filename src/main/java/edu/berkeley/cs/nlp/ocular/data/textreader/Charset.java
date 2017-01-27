package edu.berkeley.cs.nlp.ocular.data.textreader;

import static edu.berkeley.cs.nlp.ocular.util.CollectionHelper.makeMap;
import static edu.berkeley.cs.nlp.ocular.util.CollectionHelper.makeSet;
import static edu.berkeley.cs.nlp.ocular.util.CollectionHelper.setUnion;
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
import tberg.murphy.indexer.Indexer;

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

	private static boolean isPunctuation(char c) {
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

	private static boolean isCombiningChar(String c) {
		return (("\u0300".compareTo(c) <= 0 && c.compareTo("\u036F") <= 0) || 
				("\u1AB0".compareTo(c) <= 0 && c.compareTo("\u1AFF") <= 0) || 
				("\u1DC0".compareTo(c) <= 0 && c.compareTo("\u1DFF") <= 0) || 
				("\u20D0".compareTo(c) <= 0 && c.compareTo("\u20FF") <= 0) || 
				("\uFE20".compareTo(c) <= 0 && c.compareTo("\uFE2F") <= 0));
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

	private static final HashMap<String,String> COMBINING_TO_ESCAPE_MAP = new HashMap<String,String>();
	static {
		COMBINING_TO_ESCAPE_MAP.put(GRAVE_COMBINING, GRAVE_ESCAPE);
		COMBINING_TO_ESCAPE_MAP.put(ACUTE_COMBINING, ACUTE_ESCAPE);
		COMBINING_TO_ESCAPE_MAP.put(CIRCUMFLEX_COMBINING, CIRCUMFLEX_ESCAPE);
		COMBINING_TO_ESCAPE_MAP.put(TILDE_COMBINING, TILDE_ESCAPE);
		COMBINING_TO_ESCAPE_MAP.put(MACRON_COMBINING, MACRON_ESCAPE);
		COMBINING_TO_ESCAPE_MAP.put(BREVE_COMBINING, BREVE_ESCAPE);
		COMBINING_TO_ESCAPE_MAP.put(DIAERESIS_COMBINING, DIAERESIS_ESCAPE);
		COMBINING_TO_ESCAPE_MAP.put(CEDILLA_COMBINING, CEDILLA_ESCAPE);
		COMBINING_TO_ESCAPE_MAP.put(MACRON_BELOW_COMBINING, MACRON_BELOW_ESCAPE);
	}
	
//	private static String combiningToEscape(String combiningChar) {
//		String escape = COMBINING_TO_ESCAPE_MAP.get(combiningChar);
//		if (escape != null)
//			return escape;
//		else
//			throw new RuntimeException("Unrecognized combining char: [" + combiningChar + "] (" + StringHelper.toUnicode(combiningChar) + ")");
//	}

	private static String escapeToCombining(String escSeq) {
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

	private static final Map<String, String> PRECOMPOSED_TO_ESCAPED_MAP = new HashMap<String, String>();
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
		PRECOMPOSED_TO_ESCAPED_MAP.put("ĭ", "\\vi"); // \vi
		//PRECOMPOSED_TO_ESCAPED_MAP.put("ı", "\\ii"); // \ii

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

	private static final Map<String, String> PRECOMPOSED_TO_COMBINED_MAP = new HashMap<String, String>();
	static {
		for (Map.Entry<String, String> entry : PRECOMPOSED_TO_ESCAPED_MAP.entrySet()) {
			String value = entry.getValue();
			String baseChar = value.substring(value.length() - 1);
			String escapeCodes = value.substring(0, value.length() - 1);
			if (escapeCodes.length() % 2 != 0) throw new RuntimeException("problem with precomposed mapping: " + value);
			StringBuilder baseWithCombining = new StringBuilder(baseChar);
			for (int i = escapeCodes.length() - 2; i >= 0; i -= 2)
				baseWithCombining.append(escapeToCombining(escapeCodes.substring(i, i + 2)));
			PRECOMPOSED_TO_COMBINED_MAP.put(entry.getKey(), baseWithCombining.toString());
		}
	}
	
	private static final Map<String, String> COMBINED_TO_PRECOMPOSED_MAP = new HashMap<String, String>();
	static {
		for (Map.Entry<String, String> entry : PRECOMPOSED_TO_COMBINED_MAP.entrySet()) {
			COMBINED_TO_PRECOMPOSED_MAP.put(entry.getValue(), entry.getKey());
		}
	}
	
	public static final Set<String> CHARS_THAT_CAN_BE_REPLACED = setUnion(LOWERCASE_LATIN_LETTERS, makeSet("ç")); // TODO: Change this?
	public static final Set<String> VALID_CHAR_SUBSTITUTIONS = LOWERCASE_LATIN_LETTERS; // TODO: Change this?
	public static final Set<String> CHARS_THAT_CAN_DOUBLED = LOWERCASE_LATIN_LETTERS; // TODO: Change this?
	public static final Set<String> CHARS_THAT_CAN_BE_DECORATED_WITH_AN_ELISION_TILDE = LOWERCASE_LATIN_LETTERS; // TODO: Change this?
	public static final Set<String> CHARS_THAT_CAN_BE_ELIDED = LOWERCASE_LATIN_LETTERS; // TODO: Change this?
	private static final Set<String> COMBINING_DIACRITICS_THAT_CAN_BE_DISREGARDED = makeSet(GRAVE_COMBINING, ACUTE_COMBINING);
	public static final Set<String> LETTERS_WITH_DISREGARDEDABLE_DIACRITICS = LOWERCASE_VOWELS;
	
	public static Set<Integer> makePunctSet(Indexer<String> charIndexer) {
		Set<Integer> punctSet = new HashSet<Integer>();
		for (String c : charIndexer.getObjects()) {
			if (isPunctuationChar(c))
				punctSet.add(charIndexer.getIndex(c));
		}
		return punctSet;
	}
	public static Set<Integer> makeCanBeReplacedSet(Indexer<String> charIndexer) {
		Set<Integer> canBeReplaced = new HashSet<Integer>();
		for (String c : charIndexer.getObjects()) {
			if (CHARS_THAT_CAN_BE_REPLACED.contains(c))
				canBeReplaced.add(charIndexer.getIndex(c));
		}
		return canBeReplaced;
	}
	public static Set<Integer> makeValidSubstitutionCharsSet(Indexer<String> charIndexer) {
		Set<Integer> validSubstitutionChars = new HashSet<Integer>();
		for (String c : charIndexer.getObjects()) {
			if (VALID_CHAR_SUBSTITUTIONS.contains(c))
				validSubstitutionChars.add(charIndexer.getIndex(c));
		}
		return validSubstitutionChars;
	}
	public static Set<Integer> makeValidDoublableSet(Indexer<String> charIndexer) {
		Set<Integer> validDoublableChars = new HashSet<Integer>();
		for (String c : charIndexer.getObjects()) {
			if (CHARS_THAT_CAN_DOUBLED.contains(c))
				validDoublableChars.add(charIndexer.getIndex(c));
		}
		return validDoublableChars;
	}
	public static Set<Integer> makeCanBeElidedSet(Indexer<String> charIndexer) {
		Set<Integer> canBeElided = new HashSet<Integer>();
		for (String c : charIndexer.getObjects()) {
			if (CHARS_THAT_CAN_BE_ELIDED.contains(c))
				canBeElided.add(charIndexer.getIndex(c));
		}
		return canBeElided;
	}
	public static Map<Integer,Integer> makeAddTildeMap(Indexer<String> charIndexer) {
		Map<Integer,Integer> m = new HashMap<Integer, Integer>();
		for (String original : charIndexer.getObjects()) {
			Tuple2<String,List<String>> originalLetterAndCombiningDiacritics = normalizeCharSeparateDiacritics(original);
			String baseLetter = originalLetterAndCombiningDiacritics._1;
			if (CHARS_THAT_CAN_BE_DECORATED_WITH_AN_ELISION_TILDE.contains(original)) {
					m.put(charIndexer.getIndex(original), charIndexer.getIndex(addTilde(baseLetter)));
			}
			else if (LETTERS_WITH_DISREGARDEDABLE_DIACRITICS.contains(baseLetter)) {
				for (String diacritic : originalLetterAndCombiningDiacritics._2) {
					if (COMBINING_DIACRITICS_THAT_CAN_BE_DISREGARDED.contains(diacritic)) {
						m.put(charIndexer.getIndex(original), charIndexer.getIndex(addTilde(baseLetter)));
						break;
					}
				}
			}
		}
		return m;
	}
	public static Map<Integer,List<Integer>> makeLigatureMap(Indexer<String> charIndexer) {
		Map<Integer,List<Integer>> m = new HashMap<Integer, List<Integer>>();
		for (Map.Entry<String,String> entry : LIGATURES.entrySet()) {
			List<String> ligature = readNormalizeCharacters(entry.getKey());
			if (ligature.size() > 1) throw new RuntimeException("Ligature ["+entry.getKey()+"] has more than one character: "+ligature);
			List<Integer> l = new ArrayList<Integer>();
			for (String c : readNormalizeCharacters(entry.getValue()))
				l.add(charIndexer.getIndex(c));
			m.put(charIndexer.getIndex(ligature.get(0)), l);
		}
		return m;
	}
	public static Map<Integer,Integer> makeDiacriticDisregardMap(Indexer<String> charIndexer) {
		Map<Integer,Integer> m = new HashMap<Integer,Integer>();
		for (String original : charIndexer.getObjects()) { // find accented letters
			Tuple2<String,List<String>> originalLetterAndCombiningDiacritics = normalizeCharSeparateDiacritics(original);
			String baseLetter = originalLetterAndCombiningDiacritics._1;
			if (LETTERS_WITH_DISREGARDEDABLE_DIACRITICS.contains(baseLetter)) {
				for (String diacritic : originalLetterAndCombiningDiacritics._2) {
					if (COMBINING_DIACRITICS_THAT_CAN_BE_DISREGARDED.contains(diacritic)) {
						m.put(charIndexer.getIndex(original), charIndexer.getIndex(baseLetter));
						break;
					}
				}
			}
		}
		return m;
	}
	
	public static String addTilde(String c) {
		return normalizeChar(c + TILDE_COMBINING);
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
	 * @param c	A single character, potentially with diacritics encoded in any 
	 * form (composed, precomposed, escaped).
	 * @return	A string representing a single fully-escaped character, with all 
	 * diacritics (combining and precomposed) converted to their equivalent escape 
	 * sequences.
	 * @throws RuntimeException if the parameter `s` does not represent a single
	 * (potentially composed or escaped) character.
	 */
	public static String normalizeChar(String c) {
		Tuple2<String, List<String>> letterAndDiacritics = normalizeCharSeparateDiacritics(c);
		return letterAndDiacritics._1 + StringHelper.join(letterAndDiacritics._2);
	}

	/**
	 * @see edu.berkeley.cs.nlp.ocular.data.textreader.textreader.Charset.normalizeChar
	 * 
	 * @param c	A single character, potentially with diacritics encoded in any 
	 * form (composed, precomposed, escaped).
	 * @return	A fully-normalized character, with all diacritics (combining and 
	 * precomposed) converted to their equivalent normalized forms and placed in
	 * a list to be returned with the bare letter.
	 * @throws RuntimeException if the parameter `s` does not represent a single
	 * (potentially composed or escaped) character.
	 */
	public static Tuple2<String,List<String>> normalizeCharSeparateDiacritics(String c) {
		Tuple3<String, List<String>, Integer> letterAndLength = readLetterAndNormalDiacriticsAt(c, 0);
		int length = letterAndLength._3;
		if (c.length() != length) throw new RuntimeException("Could not escape ["+c+"] because it contains more than one character ("+StringHelper.toUnicode(c)+")");
		return Tuple2(letterAndLength._1, letterAndLength._2);
	}

	/**
	 * Read a single character from the line, starting at the given offset.
	 * 
	 * @see edu.berkeley.cs.nlp.ocular.data.textreader.textreader.Charset.normalizeChar
	 * 
	 * @param line	A line of text possibly containing characters with diacritics
	 * composed, precomposed, or escaped.
	 * @param offset	The offset point in `line` from which to start reading for a 
	 * character.
	 * @return	A fully-normalized character string, with all diacritics (combining
	 * and precomposed) converted to their equivalent combining forms.  Also 
	 * return the length in the ORIGINAL string of the span used to produce this 
	 * normalized character (to use as an offset when scanning through the string).
	 */
	private static Tuple2<String, Integer> readNormalizeCharAt(String line, int offset) {
		Tuple3<String, List<String>, Integer> result = readLetterAndNormalDiacriticsAt(line, offset);
		String c = result._1 + StringHelper.join(result._2);
		int length = result._3;
		return Tuple2(c, length);
	}
	
	/**
	 * Read a single character from the line including a list of all its diacritics, 
	 * starting at the given offset.
	 * 
	 * @see edu.berkeley.cs.nlp.ocular.data.textreader.textreader.Charset.normalizeChar
	 * 
	 * @param line	A line of text possibly containing characters with diacritics
	 * composed, precomposed, or normalized.
	 * @param offset	The offset point in `line` from which to start reading for a 
	 * character.
	 * @return	A fully-normalized character, with all diacritics (combining and 
	 * precomposed) converted to their equivalent combining forms and put in a list,
	 * the base letter with all diacritics removed, and the length in the ORIGINAL 
	 * string of the span used to produce this normalized character (to use as an 
	 * offset when scanning through the string).
	 */
	private static Tuple3<String, List<String>, Integer> readLetterAndNormalDiacriticsAt(String line, int offset) {
		int lineLen = line.length();
		if (offset >= lineLen) throw new RuntimeException("offset must be less than the line length");
		
		if (lineLen - offset >= 2 && line.substring(offset, offset + 2).equals("\\\\"))
			return Tuple3("\\\\", (List<String>)new ArrayList<String>(), 2); // "\\" is its own character (for "\"), not an escaped diacritic
		
		List<String> escapeDiacritics = new ArrayList<String>(); // in reversed order!
		List<String> combiningDiacritics = new ArrayList<String>();

		// get any escape prefixes characters
		int i = offset;
		while (i < lineLen && line.charAt(i) == '\\') {
			if (i + 1 >= lineLen) throw new RuntimeException("expected more after escape symbol, but found nothing: " + i + "," + lineLen + " " + line.substring(Math.max(0, i - 10), i) + "[" + line.substring(i) + "]");
			String escape = line.substring(i, i + 2);
			escapeDiacritics.add(0, escape);
			i += 2; // accept the 2-character escape sequence
		}

		if (i >= lineLen) throw new RuntimeException("expected a letter after escape code, but found nothing: " + i + "," + lineLen + " " + line.substring(Math.max(0, i - 50), i) + "[" + line.substring(i) + "]");
		String letter = String.valueOf(line.charAt(i));
		if (isCombiningChar(letter)) throw new RuntimeException("found unexpected combining char: " + i + "," + lineLen + " " + line.substring(Math.max(0, i - 50), i) + "[" + line.substring(i) + "]");
		i += 1; // accept the letter itself

		// get any combining characters
		while (i < lineLen) {
			String next = line.substring(i, i + 1);
			if (!isCombiningChar(next)) break;
			combiningDiacritics.add(next);
			i++; // accept the combining character
		}

		String deprecomposedChar = PRECOMPOSED_TO_COMBINED_MAP.get(letter);
		String letterOnly;
		if (deprecomposedChar == null) {
			letterOnly = letter;
		}
		else {
			letterOnly = String.valueOf(deprecomposedChar.charAt(0));
			for (int j = 1; j < deprecomposedChar.length(); ++j)
				combiningDiacritics.add(0, String.valueOf(deprecomposedChar.charAt(j)));
		}
		
		for (String diacritic : escapeDiacritics) {
			if (diacritic.equals("\\i")) {
				if (!letterOnly.equals("i")) throw new RuntimeException("the \\i escape sequence can only be used on the character 'i' (to indicate a no-dot i)");
				letterOnly = "ı";
			}
			else {
				combiningDiacritics.add(escapeToCombining(diacritic));
			}
		}
		
		if (letterOnly.length() != 1) throw new RuntimeException("base letter should be length 1, found: " + letterOnly);
		if (!combiningDiacritics.isEmpty()) {
			char letterChar = letterOnly.charAt(0);
			if (!(Character.isAlphabetic(letterChar))) 
				throw new RuntimeException("because there were diacritics, letter is expected, but something else was found: " + i + "," + lineLen + " " + line.substring(Math.max(0, i - 50), i) + "[" + line.substring(i) + "]");
		}
		
		return Tuple3(letterOnly, combiningDiacritics, i - offset);
	}

	/**
	 * Convert a string into a sequence of diacritic-normalized characters.
	 * 
	 * @see edu.berkeley.cs.nlp.ocular.data.textreader.textreader.Charset.normalizeChar
	 * 
	 * @param line	A line of text possibly containing characters with diacritics
	 * composed, precomposed, or escaped.
	 * @return	A fully-normalized character string, with all diacritics (combining
	 * and precomposed) converted to their equivalent combining chars.
	 */
	public static List<String> readNormalizeCharacters(String line) {
		List<String> normalizedChars = new ArrayList<String>();
		int i = 0;
		while (i < line.length()) {
			Tuple2<String, Integer> normalizedCharAndLength = readNormalizeCharAt(line, i);
			String c = normalizedCharAndLength._1;
			int length = normalizedCharAndLength._2;
			normalizedChars.add(c);
			i += length; // advance to the next character
		}
		return normalizedChars;
	}
	
	/**
	 * Convert character into unicode precomposed and combining characters
	 */
	public static String unescapeChar(String c, boolean precomposedOnly) {
		if (c.equals("\\\\")) return "\\";
		
		Tuple2<String,List<String>> letterAndNormalDiacritics = normalizeCharSeparateDiacritics(c); // use combining chars only (and make sure it's a valid character)
		String baseLetter = letterAndNormalDiacritics._1;
		List<String> diacritics = letterAndNormalDiacritics._2;
		
		if (diacritics.isEmpty()) return baseLetter;
		
		StringBuilder b = new StringBuilder();
		
		// Attempt to make a precomposed letter, falling back to composed otherwise
		String firstDiacritic = diacritics.get(0);
		String precomposed = COMBINED_TO_PRECOMPOSED_MAP.get(baseLetter + firstDiacritic); 
		if (precomposed != null)
			b.append(precomposed);
		else {
			b.append(baseLetter);
			if (!precomposedOnly) b.append(firstDiacritic);
		}

		if (precomposedOnly) {
			// Handle the rest of the diacritics
			for (int i = (precomposed != null ? 1 : 0); i < diacritics.size(); ++i) {
				String escape = COMBINING_TO_ESCAPE_MAP.get(diacritics.get(i));
				if (escape != null)
					b.insert(0, escape);
				else
					b.append(StringHelper.toUnicode(diacritics.get(i)));
			}
		}
		else {
			// Handle the rest of the diacritics
			for (int i = 1; i < diacritics.size(); ++i) {
				b.append(diacritics.get(i));
			}
		}
		
		return b.toString();
	}

	/**
	 * Convert character into unicode precomposed and combining characters
	 */
	public static String unescapeChar(String c) {
		return unescapeChar(c, false);
	}

	/**
	 * Convert character into a base character and explicit escape sequences
	 */
	public static String fullyEscapeChar(String c) {
		if (c.equals("\\\\")) return c;
		
		Tuple2<String,List<String>> letterAndNormalDiacritics = normalizeCharSeparateDiacritics(c); // use combining chars only (and make sure it's a valid character)
		String baseLetter = letterAndNormalDiacritics._1;
		List<String> diacritics = letterAndNormalDiacritics._2;
		if (baseLetter.equals("ı"))
			baseLetter = "\\ii";
		
		if (diacritics.isEmpty()) return baseLetter;
		
		StringBuilder b = new StringBuilder(baseLetter);

		// Handle the rest of the diacritics
		for (int i = 0; i < diacritics.size(); ++i) {
			String escape = COMBINING_TO_ESCAPE_MAP.get(diacritics.get(i));
			if (escape != null)
				b.insert(0, escape);
			else
				b.append(StringHelper.toUnicode(diacritics.get(i)));
		}
		
		return b.toString();
	}

	public static String removeAnyDiacriticFromChar(String c) {
		return normalizeCharSeparateDiacritics(c)._1;
	}

}
