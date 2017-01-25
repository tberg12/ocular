package edu.berkeley.cs.nlp.ocular.data.textreader;

import static edu.berkeley.cs.nlp.ocular.data.textreader.Charset.DIAERESIS_COMBINING;
import static edu.berkeley.cs.nlp.ocular.data.textreader.Charset.MACRON_COMBINING;
import static org.junit.Assert.assertEquals;

import java.util.Arrays;
import java.util.List;

import org.junit.Test;

/**
 * @author Dan Garrette (dhgarrette@gmail.com)
 */
public class RemoveAllDiacriticsTextReaderTests {

	private String s1 = "ing th\\~q || | follies of thõsè, who éither ``sæek'' out th\\\"os\\`e wæys \"and\" means, which either are sq̃uccess lessons";

	@Test
	public void test_readCharacters_qtilde_nodia() {
		TextReader tr = new RemoveAllDiacriticsTextReader(new BasicTextReader());
		assertEquals(Arrays.asList("t", "h", "q", "r"), tr.readCharacters("thq̃r"));
		assertEquals(Arrays.asList("t", "h", "q", "r"), tr.readCharacters("th\\~qr"));
	}

	@Test
	public void test_readCharacters_stackedDiacritics_nodia() {
		TextReader tr = new RemoveAllDiacriticsTextReader(new BasicTextReader());
		assertEquals(Arrays.asList("n"), tr.readCharacters("\\`\\'ñ" + MACRON_COMBINING + DIAERESIS_COMBINING));
	}

	@Test
	public void test_readCharacters_plain() {
		TextReader tr = new RemoveAllDiacriticsTextReader(new BasicTextReader());
		//assertEquals(Arrays.asList(), tr.readCharacters("tiquinhu\\-almoqu\\-ixtililia"));

		List<String> r = Arrays.asList("i", "n", "g", " ", "t", "h", "q", " ", "|", "|", " ", "|", " ", "f", "o", "l", "l", "i", "e", "s", " ", "o", "f", " ", "t", "h", "o", "s", "e", ",", " ", "w", "h", "o", " ", "e", "i", "t", "h", "e", "r", " ", "\"", "s", "æ", "e", "k", "\"", " ", "o", "u", "t", " ", "t", "h", "o", "s", "e", " ", "w", "æ", "y", "s", " ", "\"", "a", "n", "d", "\"", " ", "m", "e", "a", "n", "s", ",", " ", "w", "h", "i", "c", "h", " ", "e", "i", "t", "h", "e", "r", " ", "a", "r", "e", " ", "s", "q", "u", "c", "c", "e", "s", "s", " ", "l", "e", "s", "s", "o", "n", "s");
		assertEquals(r, tr.readCharacters(s1));

	}

}
