package edu.berkeley.cs.nlp.ocular.data.textreader;

import static edu.berkeley.cs.nlp.ocular.util.CollectionHelper.makeList;
import static edu.berkeley.cs.nlp.ocular.util.Tuple2.Tuple2;
import static org.junit.Assert.assertEquals;

import java.util.List;

import org.junit.Test;

import edu.berkeley.cs.nlp.ocular.util.StringHelper;

/**
 * @author Dan Garrette (dhgarrette@gmail.com)
 */
public class ReplaceSomeTextReaderTests {

	@Test
	public void test_readCharacters_1() {
		TextReader tr = new ReplaceSomeTextReader(makeList(Tuple2(Tuple2((List<String>) makeList("a", "b"), (List<String>) makeList("x", "y", "z")), 3)), new BasicTextReader());
		assertEquals("ab1ab2xyz3ab4ab5xyz6ab7ab8", StringHelper.join(tr.readCharacters("ab1ab2ab3ab4ab5ab6ab7ab8")));
	}

	@Test
	public void test_readCharacters_2() {
		TextReader tr = new ReplaceSomeTextReader(makeList(Tuple2(Tuple2((List<String>) makeList("a", "b"), (List<String>) makeList("x", "y", "z")), 4)), new BasicTextReader());
		assertEquals("ab1ab2ab3xyz4ab5ab6ab7xyz8", StringHelper.join(tr.readCharacters("ab1ab2ab3ab4ab5ab6ab7ab8")));
	}

	@Test
	public void test_readCharacters_3() {
		TextReader tr = new ReplaceSomeTextReader(makeList(Tuple2(Tuple2((List<String>) makeList("a", "b"), (List<String>) makeList("x", "y", "z")), 1)), new BasicTextReader());
		assertEquals("xyz", StringHelper.join(tr.readCharacters("ab")));
	}

	@Test
	public void test_readCharacters_4() {
		TextReader tr = new ReplaceSomeTextReader(makeList(Tuple2(Tuple2((List<String>) makeList("a", "b"), (List<String>) makeList("x", "y", "z")), 4)), new BasicTextReader());
		assertEquals("ab1ab2ab3xyz4ab5ab6ab7xyz", StringHelper.join(tr.readCharacters("ab1ab2ab3ab4ab5ab6ab7ab")));
	}

	@Test
	public void test_readCharacters_5() {
		TextReader tr = new ReplaceSomeTextReader(makeList( //
				Tuple2(Tuple2((List<String>) makeList("a", "b"), (List<String>) makeList("x", "y", "z")), 3), //
				Tuple2(Tuple2((List<String>) makeList("y", "z"), (List<String>) makeList("e")), 2)), // 
				new BasicTextReader());
		assertEquals("ab1ab2xyz3ab4ab5xe6ab7ab8", StringHelper.join(tr.readCharacters("ab1ab2ab3ab4ab5ab6ab7ab8")));
	}

	@Test
	public void test_readCharacters_6() {
		TextReader tr = new ReplaceSomeTextReader(makeList(Tuple2(Tuple2((List<String>) makeList("x", "x"), (List<String>) makeList("a")), 1)), new BasicTextReader());
		assertEquals("aa", StringHelper.join(tr.readCharacters("xxxx")));
	}

	@Test
	public void test_readCharacters_7() {
		TextReader tr = new ReplaceSomeTextReader(makeList(Tuple2(Tuple2((List<String>) makeList("x", "x"), (List<String>) makeList("a", "x")), 1)), new BasicTextReader());
		assertEquals("axax", StringHelper.join(tr.readCharacters("xxxx")));
	}

}
