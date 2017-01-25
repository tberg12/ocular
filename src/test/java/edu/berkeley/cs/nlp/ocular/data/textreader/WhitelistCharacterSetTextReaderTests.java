package edu.berkeley.cs.nlp.ocular.data.textreader;

import static edu.berkeley.cs.nlp.ocular.data.textreader.Charset.ACUTE_COMBINING;
import static edu.berkeley.cs.nlp.ocular.data.textreader.Charset.GRAVE_COMBINING;
import static org.junit.Assert.assertEquals;

import java.util.Arrays;

import org.junit.Test;

import edu.berkeley.cs.nlp.ocular.util.CollectionHelper;

/**
 * @author Dan Garrette (dhgarrette@gmail.com)
 */
public class WhitelistCharacterSetTextReaderTests {

	@Test
	public void test_readCharacters_default() {
		String s = "thi&s thá$t t$hè";
		WhitelistCharacterSetTextReader tr1 = new WhitelistCharacterSetTextReader(CollectionHelper.makeSet("a", "e", "h", "i", "s", "t"), new BasicTextReader());
		assertEquals(Arrays.asList("t", "h", "i", "s", " ", "t", "h", "t", " ", "t", "h"), tr1.readCharacters(s));
		WhitelistCharacterSetTextReader tr2 = new WhitelistCharacterSetTextReader(CollectionHelper.makeSet("a", "e", "h", "i", "s", "t", "\\'a"), new BasicTextReader());
		assertEquals(Arrays.asList("t", "h", "i", "s", " ", "t", "h", "a" + ACUTE_COMBINING, "t", " ", "t", "h"), tr2.readCharacters(s));
		WhitelistCharacterSetTextReader tr3 = new WhitelistCharacterSetTextReader(CollectionHelper.makeSet("a", "e", "h", "í", "s", "t"), new BasicTextReader());
		assertEquals(Arrays.asList("t", "h", "s", " ", "t", "h", "t", " ", "t", "h"), tr3.readCharacters(s));
	}

	@Test
	public void test_readCharacters_considerDiacritics() {
		String s = "thi&s thá$t t$hè";
		WhitelistCharacterSetTextReader tr1 = new WhitelistCharacterSetTextReader(CollectionHelper.makeSet("a", "e", "h", "i", "s", "t"), false, new BasicTextReader());
		assertEquals(Arrays.asList("t", "h", "i", "s", " ", "t", "h", "t", " ", "t", "h"), tr1.readCharacters(s));
		WhitelistCharacterSetTextReader tr2 = new WhitelistCharacterSetTextReader(CollectionHelper.makeSet("a", "e", "h", "i", "s", "t", "\\'a"), false, new BasicTextReader());
		assertEquals(Arrays.asList("t", "h", "i", "s", " ", "t", "h", "a" + ACUTE_COMBINING, "t", " ", "t", "h"), tr2.readCharacters(s));
		WhitelistCharacterSetTextReader tr3 = new WhitelistCharacterSetTextReader(CollectionHelper.makeSet("a", "e", "h", "í", "s", "t"), false, new BasicTextReader());
		assertEquals(Arrays.asList("t", "h", "s", " ", "t", "h", "t", " ", "t", "h"), tr3.readCharacters(s));
	}

	@Test
	public void test_readCharacters_disregardDiacritics() {
		String s = "thi&s thá$t t$hè";
		WhitelistCharacterSetTextReader tr1 = new WhitelistCharacterSetTextReader(CollectionHelper.makeSet("a", "e", "h", "i", "s", "t"), true, new BasicTextReader());
		assertEquals(Arrays.asList("t", "h", "i", "s", " ", "t", "h", "a" + ACUTE_COMBINING, "t", " ", "t", "h", "e" + GRAVE_COMBINING), tr1.readCharacters(s));
		WhitelistCharacterSetTextReader tr2 = new WhitelistCharacterSetTextReader(CollectionHelper.makeSet("a", "e", "h", "i", "s", "t", "\\'a"), true, new BasicTextReader());
		assertEquals(Arrays.asList("t", "h", "i", "s", " ", "t", "h", "a" + ACUTE_COMBINING, "t", " ", "t", "h", "e" + GRAVE_COMBINING), tr2.readCharacters(s));
		WhitelistCharacterSetTextReader tr3 = new WhitelistCharacterSetTextReader(CollectionHelper.makeSet("a", "e", "h", "í", "s", "t"), true, new BasicTextReader());
		assertEquals(Arrays.asList("t", "h", "s", " ", "t", "h", "a" + ACUTE_COMBINING, "t", " ", "t", "h", "e" + GRAVE_COMBINING), tr3.readCharacters(s));
	}

}
