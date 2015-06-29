package edu.berkeley.cs.nlp.ocular.main;

import static edu.berkeley.cs.nlp.ocular.data.textreader.Charset.*;
import static org.junit.Assert.assertEquals;

import org.junit.Test;

public class CharsetTests {

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

		assertEquals("n", removeAnyDiacriticFromChar("ñ"));
		assertEquals("n", removeAnyDiacriticFromChar("\\~n"));
		assertEquals("q", removeAnyDiacriticFromChar("q" + TILDE_COMBINING));
		assertEquals("q", removeAnyDiacriticFromChar("\\~q"));
	}

	@Test
	public void test_escapeChar() {
		assertEquals("\\`\\'\\\"\\-\\~n", escapeChar("\\`\\'ñ" + MACRON_COMBINING + DIAERESIS_COMBINING));
		assertEquals("\\`\\'\\\"\\-\\~n", escapeChar("\\`\\'n" + TILDE_COMBINING + MACRON_COMBINING + DIAERESIS_COMBINING));
		assertEquals("\\`\\'\\\"\\-\\~q", escapeChar("\\`\\'q" + TILDE_COMBINING + MACRON_COMBINING + DIAERESIS_COMBINING));

		assertEquals("\\~n", escapeChar("ñ"));
		assertEquals("\\~n", escapeChar("\\~n"));
		assertEquals("\\~q", escapeChar("q" + TILDE_COMBINING));
		assertEquals("\\~q", escapeChar("\\~q"));
		
		assertEquals("\\'a", escapeChar("á"));
	}

}
