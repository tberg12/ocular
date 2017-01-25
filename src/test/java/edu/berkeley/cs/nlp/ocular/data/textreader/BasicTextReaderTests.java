package edu.berkeley.cs.nlp.ocular.data.textreader;

import static edu.berkeley.cs.nlp.ocular.data.textreader.Charset.ACUTE_COMBINING;
import static edu.berkeley.cs.nlp.ocular.data.textreader.Charset.DIAERESIS_COMBINING;
import static edu.berkeley.cs.nlp.ocular.data.textreader.Charset.GRAVE_COMBINING;
import static edu.berkeley.cs.nlp.ocular.data.textreader.Charset.MACRON_COMBINING;
import static edu.berkeley.cs.nlp.ocular.data.textreader.Charset.TILDE_COMBINING;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.fail;

import java.util.Arrays;
import java.util.List;

import org.junit.Test;

/**
 * @author Dan Garrette (dhgarrette@gmail.com)
 */
public class BasicTextReaderTests {

	private String s1 = "ing th\\~q || | follies of thõsè, who éither ``sæek'' out th\\\"os\\`e wæys \"and\" means, which either are sq̃uccess lessons";

	@Test
	public void test_readCharacters_qtilde() {
		TextReader tr = new BasicTextReader();
		assertEquals(Arrays.asList("q" + TILDE_COMBINING), tr.readCharacters("q̃"));
		assertEquals(Arrays.asList("t", "h", "q" + TILDE_COMBINING, "r"), tr.readCharacters("thq̃r"));
		assertEquals(Arrays.asList("t", "h", "q" + TILDE_COMBINING, "r"), tr.readCharacters("th\\~qr"));
	}

	@Test
	public void test_readCharacters_stackedDiacritics() {
		TextReader tr = new BasicTextReader();
		assertEquals(Arrays.asList("n" + TILDE_COMBINING + MACRON_COMBINING + DIAERESIS_COMBINING + ACUTE_COMBINING + GRAVE_COMBINING), tr.readCharacters("\\`\\'ñ" + MACRON_COMBINING + DIAERESIS_COMBINING));
	}

	@Test
	public void test_readCharacters_dia() {
		TextReader tr = new BasicTextReader();
		List<String> r = Arrays.asList("i", "n", "g", " ", "t", "h", "q" + TILDE_COMBINING, " ", "|", "|", " ", "|", " ", "f", "o", "l", "l", "i", "e", "s", " ", "o", "f", " ", "t", "h", "o" + TILDE_COMBINING, "s", "e" + GRAVE_COMBINING, ",", " ", "w", "h", "o", " ", "e" + ACUTE_COMBINING, "i", "t", "h", "e", "r", " ", "\"", "s", "æ", "e", "k", "\"", " ", "o", "u", "t", " ", "t", "h", "o" + DIAERESIS_COMBINING, "s", "e" + GRAVE_COMBINING, " ", "w", "æ", "y", "s", " ", "\"", "a", "n", "d", "\"", " ", "m", "e", "a", "n", "s", ",", " ", "w", "h", "i", "c", "h", " ", "e", "i", "t", "h", "e", "r", " ", "a", "r", "e", " ", "s", "q" + TILDE_COMBINING, "u", "c", "c", "e", "s", "s", " ", "l", "e", "s", "s", "o", "n", "s");
		assertEquals(r, tr.readCharacters(s1));
	}

	@Test
	public void test_readCharacters_backslash() {
		TextReader tr = new BasicTextReader();
		List<String> r = Arrays.asList("t", "h", "i", "s", "\\\\", "t", "h", "a", "t", "\\\\", "t", "h", "e", "\\\\");
		assertEquals(r, tr.readCharacters("this\\\\that\\\\the\\\\"));
		try {
			List<String> r2 = tr.readCharacters("this\\that\\the\\");
			fail("Exception expected, found: ["+r2+"]");
		} catch (RuntimeException e) {
			assertEquals("Unrecognized escape sequence: [\\t]", e.getMessage());
		}
	}

	@Test
	public void test_readCharacters_noEscapeChar() {
		BasicTextReader tr = new BasicTextReader(false);
		assertEquals(Arrays.asList("t", "h", "\\\\", "~", "q", "r", "\\\\", "\\\\", "x"), tr.readCharacters("th\\~qr\\\\x"));
	}

}
