package edu.berkeley.cs.nlp.ocular.data.textreader;

import static org.junit.Assert.assertEquals;
import static edu.berkeley.cs.nlp.ocular.data.textreader.Charset.*;

import java.util.Arrays;
import java.util.List;

import org.junit.Test;

/**
 * @author Dan Garrette (dhgarrette@gmail.com)
 */
public class ConvertLongSTextReaderTests {

	private String s1 = "ing th\\~q || | follies of thõsè, who éither ``sæek'' out th\\\"os\\`e wæys \"and\" means, which either are sq̃uccess confession asi \\\\lessons";

	@Test
	public void test_readCharacters() {
		TextReader tr = new ConvertLongSTextReader(new BasicTextReader());
		assertEquals(Arrays.asList("t", "h", "o" + TILDE_COMBINING, "ſ", "e" + GRAVE_COMBINING), tr.readCharacters("thõsè"));
		assertEquals(Arrays.asList("ſ", "i"), tr.readCharacters("si"));
		assertEquals(Arrays.asList("ſ", "i", "n"), tr.readCharacters("sin"));
		assertEquals(Arrays.asList("a", "ſ", "i"), tr.readCharacters("asi"));
		assertEquals(Arrays.asList("ſ", "s", "i"), tr.readCharacters("ssi"));
		assertEquals(Arrays.asList("a", "ſ", "s", "i"), tr.readCharacters("assi"));
		assertEquals(Arrays.asList("ſ", "s", "i", "n"), tr.readCharacters("ssin"));
		assertEquals(Arrays.asList("a", "ſ", "s", "i", "n"), tr.readCharacters("assin"));
		List<String> r = Arrays.asList("i", "n", "g", " ", "t", "h", "q" + TILDE_COMBINING, " ", "|", "|", " ", "|", " ", "f", "o", "l", "l", "i", "e", "s", " ", "o", "f", " ", "t", "h", "o" + TILDE_COMBINING, "ſ", "e" + GRAVE_COMBINING, ",", " ", "w", "h", "o", " ", "e" + ACUTE_COMBINING, "i", "t", "h", "e", "r", " ", "\"", "ſ", "æ", "e", "k", "\"", " ", "o", "u", "t", " ", "t", "h", "o" + DIAERESIS_COMBINING, "ſ", "e" + GRAVE_COMBINING, " ", "w", "æ", "y", "s", " ", "\"", "a", "n", "d", "\"", " ", "m", "e", "a", "n", "s", ",", " ", "w", "h", "i", "c", "h", " ", "e", "i", "t", "h", "e", "r", " ", "a", "r", "e", " ", "ſ", "q" + TILDE_COMBINING, "u", "c", "c", "e", "ſ", "s", " ", "c", "o", "n", "f", "e", "ſ", "s", "i", "o", "n", " ", "a", "ſ", "i", " ", "\\\\", "l", "e", "ſ", "ſ", "o", "n", "s");
		assertEquals(r, tr.readCharacters(s1));
	}

	@Test
	public void test_readCharacters_removeDia() {
		TextReader tr = new ConvertLongSTextReader(new RemoveAllDiacriticsTextReader(new BasicTextReader()));
		List<String> r = Arrays.asList("i", "n", "g", " ", "t", "h", "q", " ", "|", "|", " ", "|", " ", "f", "o", "l", "l", "i", "e", "s", " ", "o", "f", " ", "t", "h", "o", "ſ", "e", ",", " ", "w", "h", "o", " ", "e", "i", "t", "h", "e", "r", " ", "\"", "ſ", "æ", "e", "k", "\"", " ", "o", "u", "t", " ", "t", "h", "o", "ſ", "e", " ", "w", "æ", "y", "s", " ", "\"", "a", "n", "d", "\"", " ", "m", "e", "a", "n", "s", ",", " ", "w", "h", "i", "c", "h", " ", "e", "i", "t", "h", "e", "r", " ", "a", "r", "e", " ", "ſ", "q", "u", "c", "c", "e", "ſ", "s", " ", "c", "o", "n", "f", "e", "ſ", "s", "i", "o", "n", " ", "a", "ſ", "i", " ", "\\\\", "l", "e", "ſ", "ſ", "o", "n", "s");
		assertEquals(r, tr.readCharacters(s1));
	}

}
