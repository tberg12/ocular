package edu.berkeley.cs.nlp.ocular.data.textreader;

import static edu.berkeley.cs.nlp.ocular.data.textreader.Charset.*;
import static edu.berkeley.cs.nlp.ocular.util.Tuple2.Tuple2;
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
		//assertEquals("ı", unescapeChar("\\ii"));
		
		assertEquals("\\\\", unescapeChar("\\\\"));
	}

	@Test
	public void test_unescapeCharPrecomposedOnly() {
		assertEquals(GRAVE_ESCAPE + ACUTE_ESCAPE + DIAERESIS_ESCAPE + MACRON_ESCAPE + "ñ", unescapeCharPrecomposedOnly("\\`\\'ñ" + MACRON_COMBINING + DIAERESIS_COMBINING));
		assertEquals(GRAVE_ESCAPE + ACUTE_ESCAPE + DIAERESIS_ESCAPE + MACRON_ESCAPE + "ñ", unescapeCharPrecomposedOnly("\\`\\'n" + TILDE_COMBINING + MACRON_COMBINING + DIAERESIS_COMBINING));
		assertEquals(GRAVE_ESCAPE + ACUTE_ESCAPE + DIAERESIS_ESCAPE + MACRON_ESCAPE + TILDE_ESCAPE + "q", unescapeCharPrecomposedOnly("\\`\\'q" + TILDE_COMBINING + MACRON_COMBINING + DIAERESIS_COMBINING));

		assertEquals("ñ", unescapeCharPrecomposedOnly("ñ"));
		assertEquals("ñ", unescapeCharPrecomposedOnly("\\~n"));
		assertEquals("\\~q", unescapeCharPrecomposedOnly("q" + TILDE_COMBINING));
		assertEquals("\\~q", unescapeCharPrecomposedOnly("\\~q"));
		//assertEquals("ı", unescapeCharPrecomposedOnly("\\ii"));
		
		assertEquals("\\\\", unescapeCharPrecomposedOnly("\\\\"));
	}

	@Test
	public void test_escapeCharSeparateDiacritics() {
		assertEquals(asList(GRAVE_ESCAPE, ACUTE_ESCAPE, DIAERESIS_ESCAPE, MACRON_ESCAPE, TILDE_ESCAPE), escapeCharSeparateDiacritics("\\`\\'ñ" + MACRON_COMBINING + DIAERESIS_COMBINING)._1);
		assertEquals(asList(GRAVE_ESCAPE, ACUTE_ESCAPE, DIAERESIS_ESCAPE, MACRON_ESCAPE, TILDE_ESCAPE), escapeCharSeparateDiacritics("\\`\\'n" + TILDE_COMBINING + MACRON_COMBINING + DIAERESIS_COMBINING)._1);
		assertEquals(asList(GRAVE_ESCAPE, ACUTE_ESCAPE, DIAERESIS_ESCAPE, MACRON_ESCAPE, TILDE_ESCAPE), escapeCharSeparateDiacritics("\\`\\'q" + TILDE_COMBINING + MACRON_COMBINING + DIAERESIS_COMBINING)._1);

		assertEquals(asList(), escapeCharSeparateDiacritics("t")._1);
		assertEquals(asList(TILDE_ESCAPE), escapeCharSeparateDiacritics("ñ")._1);
		assertEquals(asList(TILDE_ESCAPE), escapeCharSeparateDiacritics("\\~n")._1);
		assertEquals(asList(TILDE_ESCAPE), escapeCharSeparateDiacritics("q̃")._1);
		assertEquals(asList(TILDE_ESCAPE), escapeCharSeparateDiacritics("q" + TILDE_COMBINING)._1);
		assertEquals(asList(TILDE_ESCAPE), escapeCharSeparateDiacritics("\\~q")._1);
		assertEquals(asList(), escapeCharSeparateDiacritics("\\\\")._1);

		assertEquals("n", escapeCharSeparateDiacritics("\\`\\'ñ" + MACRON_COMBINING + DIAERESIS_COMBINING)._2);
		assertEquals("n", escapeCharSeparateDiacritics("\\`\\'n" + TILDE_COMBINING + MACRON_COMBINING + DIAERESIS_COMBINING)._2);
		assertEquals("q", escapeCharSeparateDiacritics("\\`\\'q" + TILDE_COMBINING + MACRON_COMBINING + DIAERESIS_COMBINING)._2);

		assertEquals("t", escapeCharSeparateDiacritics("t")._2);
		assertEquals("n", escapeCharSeparateDiacritics("ñ")._2);
		assertEquals("n", escapeCharSeparateDiacritics("\\~n")._2);
		assertEquals("q", escapeCharSeparateDiacritics("q̃")._2);
		assertEquals("q", escapeCharSeparateDiacritics("q" + TILDE_COMBINING)._2);
		assertEquals("q", escapeCharSeparateDiacritics("\\~q")._2);
		assertEquals("\\\\", escapeCharSeparateDiacritics("\\\\")._2);
		
		try {
			Tuple2<List<String>,String> r = escapeCharSeparateDiacritics(MACRON_ESCAPE + "" + TILDE_COMBINING);
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
	public void test_escapeChar() {
		assertEquals("t", escapeChar("t"));
		assertEquals("\\~q", escapeChar("q̃"));
		assertEquals("\\~q", escapeChar("q" + TILDE_COMBINING));
		assertEquals("\\~q", escapeChar("\\~q"));
		assertEquals("\\~n", escapeChar("ñ"));
		assertEquals("\\~n", escapeChar("\\~n"));
		assertEquals("\\'a", escapeChar("á"));

		assertEquals("\\`\\'\\\"\\-\\~n", escapeChar("\\`\\'ñ" + MACRON_COMBINING + DIAERESIS_COMBINING));
		assertEquals("\\`\\'\\\"\\-\\~n", escapeChar("\\`\\'n" + TILDE_COMBINING + MACRON_COMBINING + DIAERESIS_COMBINING));
		assertEquals("\\`\\'\\\"\\-\\~q", escapeChar("\\`\\'q" + TILDE_COMBINING + MACRON_COMBINING + DIAERESIS_COMBINING));
		
		assertEquals("\\\\", escapeChar("\\\\"));
	}

	@Test
	public void test_readCharAt() {
		//String s1 = "ing th\\~q || | follies of thõsè, who éither ``sæek'' out th\\\"os\\`e wæys \"and\" means, which either are sq̃uccess lessons";
		assertEquals(Tuple2("\\\\", 2), readCharAt("this\\\\that", 4));
	}

}
