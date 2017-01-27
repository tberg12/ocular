package edu.berkeley.cs.nlp.ocular.data.textreader;

import static edu.berkeley.cs.nlp.ocular.data.textreader.Charset.*;
import static java.util.Arrays.asList;
import static org.junit.Assert.*;

import java.util.List;

import org.junit.Test;

import edu.berkeley.cs.nlp.ocular.util.Tuple2;

/**
 * @author Dan Garrette (dhgarrette@gmail.com)
 */
public class CharsetTests {

	@Test
	public void test_isPunctuationChar() {
		assertFalse(isPunctuationChar("t"));
		assertFalse(isPunctuationChar("q̃"));
		assertFalse(isPunctuationChar("\\~q"));
		assertFalse(isPunctuationChar("\\`\\'ñ" + MACRON_COMBINING + DIAERESIS_COMBINING));
		assertTrue(isPunctuationChar(";"));
		assertTrue(isPunctuationChar("\\\\"));
		try { isPunctuationChar(";;"); fail("no exception thrown"); } catch (RuntimeException e) { e.getMessage().contains("contains more than one character"); }
	}

	@Test
	public void test_unescapeChar() {
		assertEquals("ñ" + MACRON_COMBINING + DIAERESIS_COMBINING + ACUTE_COMBINING + GRAVE_COMBINING, unescapeChar("\\`\\'ñ" + MACRON_COMBINING + DIAERESIS_COMBINING));
		assertEquals("ñ" + MACRON_COMBINING + DIAERESIS_COMBINING + ACUTE_COMBINING + GRAVE_COMBINING, unescapeChar("\\`\\'n" + TILDE_COMBINING + MACRON_COMBINING + DIAERESIS_COMBINING));
		assertEquals("q" + TILDE_COMBINING + MACRON_COMBINING + DIAERESIS_COMBINING + ACUTE_COMBINING + GRAVE_COMBINING, unescapeChar("\\`\\'q" + TILDE_COMBINING + MACRON_COMBINING + DIAERESIS_COMBINING));

		assertEquals("ñ", unescapeChar("ñ"));
		assertEquals("ñ", unescapeChar("\\~n"));
		assertEquals("q" + TILDE_COMBINING, unescapeChar("q" + TILDE_COMBINING));
		assertEquals("q" + TILDE_COMBINING, unescapeChar("\\~q"));
		assertEquals("ı", unescapeChar("\\ii"));
		assertEquals("ı", unescapeChar("ı"));
		
		assertEquals("\\", unescapeChar("\\\\"));
	}

	@Test
	public void test_unescapeChar_precomposedOnly() {
		assertEquals(GRAVE_ESCAPE + ACUTE_ESCAPE + DIAERESIS_ESCAPE + MACRON_ESCAPE + "ñ", unescapeChar("\\`\\'ñ" + MACRON_COMBINING + DIAERESIS_COMBINING, true));
		assertEquals(GRAVE_ESCAPE + ACUTE_ESCAPE + DIAERESIS_ESCAPE + MACRON_ESCAPE + "ñ", unescapeChar("\\`\\'n" + TILDE_COMBINING + MACRON_COMBINING + DIAERESIS_COMBINING, true));
		assertEquals(GRAVE_ESCAPE + ACUTE_ESCAPE + DIAERESIS_ESCAPE + MACRON_ESCAPE + TILDE_ESCAPE + "q", unescapeChar("\\`\\'q" + TILDE_COMBINING + MACRON_COMBINING + DIAERESIS_COMBINING, true));

		assertEquals("ñ", unescapeChar("ñ", true));
		assertEquals("ñ", unescapeChar("\\~n", true));
		assertEquals("\\~q", unescapeChar("q" + TILDE_COMBINING, true));
		assertEquals("\\~q", unescapeChar("\\~q", true));
		assertEquals("ı", unescapeChar("\\ii", true));
		assertEquals("ı", unescapeChar("ı", true));
		
		assertEquals("\\", unescapeChar("\\\\", true));
	}

	@Test
	public void test_fullyEscapeChar() {
		assertEquals(GRAVE_ESCAPE + ACUTE_ESCAPE + DIAERESIS_ESCAPE + MACRON_ESCAPE + TILDE_ESCAPE + "n", fullyEscapeChar("\\`\\'ñ" + MACRON_COMBINING + DIAERESIS_COMBINING));
		assertEquals(GRAVE_ESCAPE + ACUTE_ESCAPE + DIAERESIS_ESCAPE + MACRON_ESCAPE + TILDE_ESCAPE + "n", fullyEscapeChar("\\`\\'n" + TILDE_COMBINING + MACRON_COMBINING + DIAERESIS_COMBINING));
		assertEquals(GRAVE_ESCAPE + ACUTE_ESCAPE + DIAERESIS_ESCAPE + MACRON_ESCAPE + TILDE_ESCAPE + "q", fullyEscapeChar("\\`\\'q" + TILDE_COMBINING + MACRON_COMBINING + DIAERESIS_COMBINING));

		assertEquals("\\~n", fullyEscapeChar("ñ"));
		assertEquals("\\~n", fullyEscapeChar("\\~n"));
		assertEquals("\\~q", fullyEscapeChar("q" + TILDE_COMBINING));
		assertEquals("\\~q", fullyEscapeChar("\\~q"));
		assertEquals("\\ii", fullyEscapeChar("\\ii"));
		assertEquals("\\ii", fullyEscapeChar("ı"));
		
		assertEquals("\\\\", fullyEscapeChar("\\\\"));
	}

	@Test
	public void test_normalizeCharSeparateDiacritics() {
		assertEquals(asList(TILDE_COMBINING, MACRON_COMBINING, DIAERESIS_COMBINING, ACUTE_COMBINING, GRAVE_COMBINING), normalizeCharSeparateDiacritics("\\`\\'ñ" + MACRON_COMBINING + DIAERESIS_COMBINING)._2);
		assertEquals(asList(TILDE_COMBINING, MACRON_COMBINING, DIAERESIS_COMBINING, ACUTE_COMBINING, GRAVE_COMBINING), normalizeCharSeparateDiacritics("\\`\\'n" + TILDE_COMBINING + MACRON_COMBINING + DIAERESIS_COMBINING)._2);
		assertEquals(asList(TILDE_COMBINING, MACRON_COMBINING, DIAERESIS_COMBINING, ACUTE_COMBINING, GRAVE_COMBINING), normalizeCharSeparateDiacritics("\\`\\'q" + TILDE_COMBINING + MACRON_COMBINING + DIAERESIS_COMBINING)._2);

		assertEquals(asList(), normalizeCharSeparateDiacritics("t")._2);
		assertEquals(asList(TILDE_COMBINING), normalizeCharSeparateDiacritics("ñ")._2);
		assertEquals(asList(TILDE_COMBINING), normalizeCharSeparateDiacritics("\\~n")._2);
		assertEquals(asList(TILDE_COMBINING), normalizeCharSeparateDiacritics("q̃")._2);
		assertEquals(asList(TILDE_COMBINING), normalizeCharSeparateDiacritics("q" + TILDE_COMBINING)._2);
		assertEquals(asList(TILDE_COMBINING), normalizeCharSeparateDiacritics("\\~q")._2);
		assertEquals(asList(), normalizeCharSeparateDiacritics("\\\\")._2);

		assertEquals("n", normalizeCharSeparateDiacritics("\\`\\'ñ" + MACRON_COMBINING + DIAERESIS_COMBINING)._1);
		assertEquals("n", normalizeCharSeparateDiacritics("\\`\\'n" + TILDE_COMBINING + MACRON_COMBINING + DIAERESIS_COMBINING)._1);
		assertEquals("q", normalizeCharSeparateDiacritics("\\`\\'q" + TILDE_COMBINING + MACRON_COMBINING + DIAERESIS_COMBINING)._1);

		assertEquals("t", normalizeCharSeparateDiacritics("t")._1);
		assertEquals("n", normalizeCharSeparateDiacritics("ñ")._1);
		assertEquals("n", normalizeCharSeparateDiacritics("\\~n")._1);
		assertEquals("q", normalizeCharSeparateDiacritics("q̃")._1);
		assertEquals("q", normalizeCharSeparateDiacritics("q" + TILDE_COMBINING)._1);
		assertEquals("q", normalizeCharSeparateDiacritics("\\~q")._1);
		assertEquals("\\\\", normalizeCharSeparateDiacritics("\\\\")._1);
		
		try {
			Tuple2<String,List<String>> r = normalizeCharSeparateDiacritics(MACRON_ESCAPE + "" + TILDE_COMBINING);
			fail("Exception expected, found: ["+r+"]");
		} catch (RuntimeException e) {
			//assertEquals("Character contains only escape codes!", e.getMessage());
		}
	}

	@Test
	public void test_removeAnyDiacriticFromChar() {
		assertEquals("n", removeAnyDiacriticFromChar("\\`\\'ñ" + MACRON_COMBINING + DIAERESIS_COMBINING));
		assertEquals("n", removeAnyDiacriticFromChar("\\`\\'n" + TILDE_COMBINING + MACRON_COMBINING + DIAERESIS_COMBINING));
		assertEquals("q", removeAnyDiacriticFromChar("\\`\\'q" + TILDE_COMBINING + MACRON_COMBINING + DIAERESIS_COMBINING));

		assertEquals("t", removeAnyDiacriticFromChar("t"));
		assertEquals("n", removeAnyDiacriticFromChar("ñ"));
		assertEquals("n", removeAnyDiacriticFromChar("\\~n"));
		assertEquals("q", removeAnyDiacriticFromChar("q̃"));
		assertEquals("q", removeAnyDiacriticFromChar("q" + TILDE_COMBINING));
		assertEquals("q", removeAnyDiacriticFromChar("\\~q"));
		assertEquals("\\\\", removeAnyDiacriticFromChar("\\\\"));
	}

	@Test
	public void test_normalizeChar() {
		assertEquals("t", normalizeChar("t"));
		assertEquals("q" + TILDE_COMBINING, normalizeChar("q̃"));
		assertEquals("q" + TILDE_COMBINING, normalizeChar("q" + TILDE_COMBINING));
		assertEquals("q" + TILDE_COMBINING, normalizeChar("\\~q"));
		assertEquals("n" + TILDE_COMBINING, normalizeChar("ñ"));
		assertEquals("n" + TILDE_COMBINING, normalizeChar("\\~n"));
		assertEquals("a" + ACUTE_COMBINING, normalizeChar("á"));
		assertEquals("ı", normalizeChar("ı"));
		assertEquals("ı", normalizeChar("\\ii"));

		assertEquals("a\u0347", normalizeChar("a\u0347"));
		
		assertEquals("n" + TILDE_COMBINING + MACRON_COMBINING + DIAERESIS_COMBINING + ACUTE_COMBINING + GRAVE_COMBINING, normalizeChar("\\`\\'ñ" + MACRON_COMBINING + DIAERESIS_COMBINING));
		assertEquals("n" + TILDE_COMBINING + MACRON_COMBINING + DIAERESIS_COMBINING + ACUTE_COMBINING + GRAVE_COMBINING, normalizeChar("\\`\\'n" + TILDE_COMBINING + MACRON_COMBINING + DIAERESIS_COMBINING));
		assertEquals("q" + TILDE_COMBINING + MACRON_COMBINING + DIAERESIS_COMBINING + ACUTE_COMBINING + GRAVE_COMBINING, normalizeChar("\\`\\'q" + TILDE_COMBINING + MACRON_COMBINING + DIAERESIS_COMBINING));
		
		assertEquals("\\\\", normalizeChar("\\\\"));
	}

	@Test
	public void test_readNormalizeCharacters() {
		assertEquals(asList("a", "b\u0311", "c", "d"), readNormalizeCharacters("ab\u0311cd"));
		assertEquals(asList("a", "b\uFE20", "c\uFE21", "d"), readNormalizeCharacters("ab\uFE20c\uFE21d"));
		assertEquals(asList("a", "b\u0361", "c", "d"), readNormalizeCharacters("ab\u0361cd"));
		assertEquals(asList("t", "a", "u\u0361", "g", "a", "a", "m"), readNormalizeCharacters("tau͡gaam"));
	}

}
