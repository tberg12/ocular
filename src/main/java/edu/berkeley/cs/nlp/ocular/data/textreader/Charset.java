package edu.berkeley.cs.nlp.ocular.data.textreader;

import static edu.berkeley.cs.nlp.ocular.util.CollectionHelper.getOrElse;
import static edu.berkeley.cs.nlp.ocular.util.CollectionHelper.makeSet;

import java.util.HashMap;
import java.util.Map;
import java.util.Set;

import tuple.Pair;
import edu.berkeley.cs.nlp.ocular.util.StringHelper;

/**
 * @author Dan Garrette (dhg@cs.utexas.edu)
 */
public class Charset {

	public static final String SPACE = " ";
	public static final String HYPHEN = "-";
	public static final String LONG_S = "|";
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
		return !Character.isAlphabetic(c) && !Character.isDigit(c);
	}
	public static boolean isPunctuation(String s) {
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
	public static final Set<String> COMBINING_CHARS = makeSet(GRAVE_COMBINING, ACUTE_COMBINING, CIRCUMFLEX_COMBINING, TILDE_COMBINING, MACRON_COMBINING, BREVE_COMBINING, DIAERESIS_COMBINING, CEDILLA_COMBINING);
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
	public static final Set<String> ESCAPE_CHARS = makeSet(GRAVE_ESCAPE, ACUTE_ESCAPE, CIRCUMFLEX_ESCAPE, TILDE_ESCAPE, MACRON_ESCAPE, BREVE_ESCAPE, DIAERESIS_ESCAPE, CEDILLA_ESCAPE);

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
		//note \_ refers to an underscore
		//note for "breve" (u over letter) mark \va
	}

	private static final Map<String, String> ESCAPED_TO_PRECOMPOSED_MAP = new HashMap<String, String>();
	static {
		for (Map.Entry<String, String> entry : PRECOMPOSED_TO_ESCAPED_MAP.entrySet())
			ESCAPED_TO_PRECOMPOSED_MAP.put(entry.getValue(), entry.getKey());
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
		Pair<String, Integer> letterAndLength = readCharAt(s, 0);
		String c = letterAndLength.getFirst();
		int length = letterAndLength.getSecond();
		if (s.length() - length != 0) throw new RuntimeException("Could not escape [" + s + "] because it contains more than one character ("+StringHelper.toUnicode(s)+")");
		return c;
	}

	/**
	 * @see edu.berkeley.cs.nlp.ocular.data.textreader.Charset.escapeChar
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
	public static Pair<String, Integer> readCharAt(String line, int offset) {
		StringBuilder sb = new StringBuilder();
		int lineLen = line.length();

		// get any escape prefixes characters
		int i = offset;
		while (i < lineLen && line.substring(i, i + 1).equals("\\")) {
			if (i + 1 >= lineLen) throw new RuntimeException("expected more after escape symbol, but found nothing: " + i + "," + lineLen + " " + line.substring(Math.max(0, i - 10), i) + "[" + line.substring(i) + "]");
			String escape = line.substring(i, i + 2);
			if (escape.equals("\\\\")) break; // "\\" is its own character (for "\"), not an escaped diacritic
			sb.append(escape);
			i += 2; // accept the 2-character escape sequence
		}

		int combiningCharCodeInsertionPoint = i - offset;

		if (i >= lineLen) throw new RuntimeException("expected a letter after escape code, but found nothing: " + i + "," + lineLen + " " + line.substring(Math.max(0, i - 50), i) + "[" + line.substring(i) + "]");
		String letter = line.substring(i, i + 1);
		i += 1; // accept the letter itself

		// get any combining characters
		while (i < lineLen) {
			String next = line.substring(i, i + 1);
			if (!isCombiningChar(next)) break;
			sb.insert(combiningCharCodeInsertionPoint, combiningToEscape(next));
			i++; // accept the combining character
		}

		sb.append(getOrElse(Charset.PRECOMPOSED_TO_ESCAPED_MAP, letter, letter)); // turn any precomposed letters into escaped letters
		return Pair.makePair(sb.toString(), i - offset);
	}
	

	/**
	 * Convert diacritic escape sequences on a character into unicode precomposed and combining characters
	 */
	public static String unescapeChar(String c) {
		String e = escapeChar(c); // use escapes only (and make sure it's a valid character)

		if (c.length() == 1) return c; // no diacritics

		StringBuilder b = new StringBuilder();
		// Attempt to make a precomposed letter, falling back to composed otherwise
		String last = e.substring(e.length() - 3); // last escape + letter
		String precomposed = ESCAPED_TO_PRECOMPOSED_MAP.get(last);
		b.append((precomposed != null ? precomposed : last.substring(2) + escapeToCombining(last.substring(0, 2))));

		for (int i = e.length() - 5; i >= 0; i -= 2) {
			String esc = e.substring(i, i + 2);
			b.append(escapeToCombining(esc));
		}
		
		return b.toString();
	}

	public static String removeAnyDiacriticFromChar(String c) {
		String escaped = escapeChar(c);
		while (escaped.charAt(0) == '\\') {
		  escaped = escaped.substring(2);
		}
		if (escaped.isEmpty()) throw new RuntimeException("Character contains only escape codes!");
		return escaped;
	}

}
