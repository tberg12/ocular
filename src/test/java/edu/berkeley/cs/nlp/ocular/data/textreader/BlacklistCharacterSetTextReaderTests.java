package edu.berkeley.cs.nlp.ocular.data.textreader;

import static org.junit.Assert.assertEquals;

import java.util.Arrays;
import java.util.List;

import org.junit.Test;

import edu.berkeley.cs.nlp.ocular.util.CollectionHelper;

/**
 * @author Dan Garrette (dhgarrette@gmail.com)
 */
public class BlacklistCharacterSetTextReaderTests {

	@Test
	public void test_readCharacters() {
		String s = "thi&s tha$t t$he";
		
		TextReader tr = new BlacklistCharacterSetTextReader(CollectionHelper.makeSet("&", "$"), new BasicTextReader());
		assertEqualsList(Arrays.asList("t", "h", "i", "s", " ", "t", "h", "a", "t", " ", "t", "h", "e"), tr.readCharacters(s));
	}

	private <A> void assertEqualsList(List<A> expected, List<A> actual) {
		assertEquals(expected.size(), actual.size());
		for (int i = 0; i < expected.size(); ++i) {
			assertEquals(expected.get(i), actual.get(i));
		}
	}
}
