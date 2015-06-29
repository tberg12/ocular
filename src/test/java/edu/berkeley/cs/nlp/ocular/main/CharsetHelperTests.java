package edu.berkeley.cs.nlp.ocular.main;

import static edu.berkeley.cs.nlp.ocular.data.textreader.Charset.*;
import static org.junit.Assert.assertEquals;

import java.util.Arrays;
import java.util.List;

import org.junit.Test;

import edu.berkeley.cs.nlp.ocular.data.textreader.CharsetHelper;

public class CharsetHelperTests {

	@Test
	public void test_splitToEncodedWords_1() {
		List<String> chars = Arrays.asList("t", "h", "í", "s", " ", "a", "n", "d", " ", ",", "t", "h", "-", "a", "t", ".");
		List<String> words = CharsetHelper.splitToEncodedWords(chars);
		assertEquals(Arrays.asList("th\\'is", "and", "th-at"), words);
	}

	@Test
	public void test_splitToEncodedWords_2() {
		List<String> chars = Arrays.asList("t", "h", "í", "s", " ", "a", "n", "d", " ", ",", "t", "h", "-", "a", "t", ".", " ");
		List<String> words = CharsetHelper.splitToEncodedWords(chars);
		assertEquals(Arrays.asList("th\\'is", "and", "th-at"), words);
	}

}
