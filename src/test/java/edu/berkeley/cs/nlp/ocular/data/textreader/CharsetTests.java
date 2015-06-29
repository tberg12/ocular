package edu.berkeley.cs.nlp.ocular.data.textreader;

import static edu.berkeley.cs.nlp.ocular.data.textreader.Charset.*;
import static edu.berkeley.cs.nlp.ocular.util.Tuple2.makeTuple2;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import org.junit.Test;

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
	}

	@Test
	public void test_readCharAt() {
		//String s1 = "ing th\\~q || | follies of thõsè, who éither ``sæek'' out th\\\"os\\`e wæys \"and\" means, which either are sq̃uccess lessons";
		assertEquals(makeTuple2("\\\\", 2), readCharAt("this\\\\that", 4));
	}

}
