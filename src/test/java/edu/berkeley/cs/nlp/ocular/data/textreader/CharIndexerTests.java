package edu.berkeley.cs.nlp.ocular.data.textreader;

import static edu.berkeley.cs.nlp.ocular.data.textreader.Charset.TILDE_COMBINING;
import static edu.berkeley.cs.nlp.ocular.data.textreader.Charset.TILDE_ESCAPE;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

import tberg.murphy.indexer.Indexer;

/**
 * @author Dan Garrette (dhgarrette@gmail.com)
 */
public class CharIndexerTests {

	@Test
	public void test() {
		Indexer<String> i = new CharIndexer();

		String ae = TILDE_ESCAPE + "a";
		String ac = "a" + TILDE_COMBINING;

		String ee = TILDE_ESCAPE + "e";
		String ec = "e" + TILDE_COMBINING;

		String ne = TILDE_ESCAPE + "n";
		String nc = "n" + TILDE_COMBINING;
		String np = "ñ";

		i.index(new String[] { "a", "b", ec });

		assertTrue(i.contains("a"));
		assertTrue(i.contains("b"));
		assertTrue(i.contains(ec));
		assertTrue(i.contains(ee));
		assertEquals(0, i.getIndex("a"));
		assertEquals("a", i.getObject(0));
		assertEquals(1, i.getIndex("b"));
		assertEquals("b", i.getObject(1));
		assertEquals(2, i.getIndex(ec));
		assertEquals(ec, i.getObject(2));
		assertEquals(2, i.getIndex(ec));
		assertEquals(3, i.size());

		assertFalse(i.contains(ae));
		assertFalse(i.contains(ac));
		assertEquals(3, i.getIndex(ae));
		assertTrue(i.contains(ae));
		assertTrue(i.contains(ac));
		assertEquals(3, i.getIndex(ac));
		assertTrue(i.contains(ae));
		assertTrue(i.contains(ac));
		assertEquals(4, i.size());

		assertFalse(i.contains(ne));
		assertFalse(i.contains(nc));
		assertFalse(i.contains(np));
		assertEquals(4, i.getIndex(np));
		assertEquals(nc, i.getObject(4));
		assertTrue(i.contains(ne));
		assertTrue(i.contains(nc));
		assertTrue(i.contains(np));
		assertEquals(4, i.getIndex(ne));
		assertEquals(4, i.getIndex(nc));
		assertEquals(nc, i.getObject(4));
		assertEquals(5, i.size());

		assertFalse(i.locked());
		i.lock();
		assertTrue(i.locked());
	}

}
