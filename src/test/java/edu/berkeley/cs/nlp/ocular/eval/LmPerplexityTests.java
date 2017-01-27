package edu.berkeley.cs.nlp.ocular.eval;

import static org.junit.Assert.*;
import tberg.murphy.indexer.HashMapIndexer;
import tberg.murphy.indexer.Indexer;

import java.util.Arrays;
import java.util.Set;

import org.junit.Test;

import edu.berkeley.cs.nlp.ocular.data.textreader.CharIndexer;
import edu.berkeley.cs.nlp.ocular.lm.CodeSwitchLanguageModel;
import edu.berkeley.cs.nlp.ocular.lm.SingleLanguageModel;
import static edu.berkeley.cs.nlp.ocular.util.CollectionHelper.intArrayToList;

/**
 * @author Hannah Alpert-Abrams (halperta@gmail.com)
 * @author Dan Garrette (dhgarrette@gmail.com)
 */
public class LmPerplexityTests {
	
	@SuppressWarnings("serial")
	@Test
	public void test_calculatePerplexity() {
		
		final CharIndexer charIndexer = new CharIndexer();
		charIndexer.index(new String[] { "a", "b", "x", "y", " " });
		
		final Indexer<String> langIndexer = new HashMapIndexer<String>();
		langIndexer.index(new String[]{ "Lang1", "Lang2" });
		
		final int a = charIndexer.getIndex("a");
		final int b = charIndexer.getIndex("b");
		final int x = charIndexer.getIndex("x");
		final int y = charIndexer.getIndex("y");
		final int s = charIndexer.getIndex(" ");
		
		final int l1 = langIndexer.getIndex("Lang1");
		final int l2 = langIndexer.getIndex("Lang2");
		
		final int[] ctx_ = new int[] {};
		final int[] ctx_a = new int[] { a };
		final int[] ctx_ab = new int[] { a , b };
		final int[] ctx_ab_ = new int[] { a , b , s };
		final int[] ctx_b_a = new int[] { b , s , a };
		final int[] ctx__ab = new int[] { s, a , b };
		final int[] ctx_x = new int[] { x };
		final int[] ctx_xy = new int[] { x , y };
		
		final SingleLanguageModel lang1Lm = new SingleLanguageModel() {
			@Override public int getMaxOrder() { return 4; }
			@Override public double getCharNgramProb(int[] context, int c) {
				if (c == a) {
					if (sameIntArray(context, ctx_)) return 0.11; 
					if (sameIntArray(context, ctx_ab_)) return 0.12; 
				}
				if (c == b) {
					if (sameIntArray(context, ctx_a)) return 0.13; 
					if (sameIntArray(context, ctx_b_a)) return 0.14; 
				}
				if (c == s) {
					if (sameIntArray(context, ctx_ab)) return 0.15; 
					if (sameIntArray(context, ctx__ab)) return 0.16; 
				}
				throw new RuntimeException("getCharNgramProb(" + intArrayToList(context) + ", " + c + ")");
			}
			@Override public Indexer<String> getCharacterIndexer() { return charIndexer; }
			@Override public Set<Integer> getActiveCharacters() { throw new RuntimeException(); }
			@Override public int[] shrinkContext(int[] originalContext) { throw new RuntimeException(); }
			@Override public boolean containsContext(int[] context) { throw new RuntimeException(); }
		};

		final SingleLanguageModel lang2Lm = new SingleLanguageModel() {
			@Override public int getMaxOrder() { return 4; }
			@Override public double getCharNgramProb(int[] context, int c) {
				if (c == x) {
					if (sameIntArray(context, ctx_)) return 0.21; 
				}
				if (c == y) {
					if (sameIntArray(context, ctx_x)) return 0.22; 
				}
				if (c == s) {
					if (sameIntArray(context, ctx_xy)) return 0.23; 
				}
				throw new RuntimeException("getCharNgramProb(" + intArrayToList(context) + ", " + c + ")");
			}
			@Override public Indexer<String> getCharacterIndexer() { return charIndexer; }
			@Override public Set<Integer> getActiveCharacters() { throw new RuntimeException(); }
			@Override public int[] shrinkContext(int[] originalContext) { throw new RuntimeException(); }
			@Override public boolean containsContext(int[] context) { throw new RuntimeException(); }
		};
		
		final CodeSwitchLanguageModel csLm = new CodeSwitchLanguageModel() {
			@Override public double getCharNgramProb(int[] context, int c) { throw new RuntimeException(); }
			@Override public Indexer<String> getCharacterIndexer() { return charIndexer; }
			@Override public Indexer<String> getLanguageIndexer() { return langIndexer; }
			@Override public SingleLanguageModel get(int language) { 
				if (language == langIndexer.getIndex("Lang1")) return lang1Lm;
				if (language == langIndexer.getIndex("Lang2")) return lang2Lm;
				throw new RuntimeException();
			}
			@Override public double languagePrior(int language)  { 
				if (language == langIndexer.getIndex("Lang1")) return 0.31;
				throw new RuntimeException();
			}
			@Override public double languageTransitionProb(int fromLanguage, int destLanguage) { 
				if (fromLanguage == langIndexer.getIndex("Lang1")) {
					if (destLanguage == langIndexer.getIndex("Lang1")) return 0.32;
					if (destLanguage == langIndexer.getIndex("Lang2")) return 0.33;
				}
				if (fromLanguage == langIndexer.getIndex("Lang2")) {
					if (destLanguage == langIndexer.getIndex("Lang1")) return 0.35;
					if (destLanguage == langIndexer.getIndex("Lang2")) return 0.34;
				}
				throw new RuntimeException();
			}
			@Override public double getProbKeepSameLanguage() { throw new RuntimeException(); }
		};

		
		LmPerplexity lmPerplexity = new LmPerplexity(csLm);

		/*
		 * "aba"
		 * 
		 * P1(a|[])    * P(L1)        =  0.11 * 0.31
		 * P1(b|[a])   * P(L1|L1)     =  0.13 * 1.00
		 * P1( |[ab])  * P(L1|L1)     =  0.15 * 1.00
		 *                               --------------------
		 *                            =  0.00066495 ^(-1/3)
		 *                            =  11.456984790348551
		 */
		double p1 = lmPerplexity.perplexity(Arrays.asList(a, b, s), Arrays.asList(l1, l1, l1));
		assertEquals(11.456984790348551, p1, 0.00000000000001);
		
		/*
		 * Lang1: a,b
		 * Lang2: x,y
		 * 
		 * "ab ab xy ab"
		 * 
		 * P1(a|[])    * P(L1)        =  0.11 * 0.31
		 * P1(b|[a])   * P(L1|L1)     =  0.13 * 1.00
		 * P1( |[ab])  * P(L1|L1)     =  0.15 * 1.00
		 * P1(a|[ab ]) * P(L1|L1)     =  0.12 * 0.32
		 * P1(b|[b a]) * P(L1|L1)     =  0.14 * 1.00
		 * P1( |[ ab]) * P(L1|L1)     =  0.16 * 1.00
		 * P2(x|[])    * P(L2|L1)     =  0.21 * 0.33
		 * P2(y|[x])   * P(L2|L2)     =  0.22 * 1.00
		 * P2( |[xy])  * P(L2|L2)     =  0.23 * 1.00
		 * P1(a|[])    * P(L1|L2)     =  0.11 * 0.35
		 * P1(b|[a])   * P(L1|L1)     =  0.13 * 1.00
		 *                               --------------------
		 *                            =  1.0038205132552398E-11 ^(-1/11)
		 *                            =  9.996534024760905
		 */
		double p2 = lmPerplexity.perplexity(Arrays.asList(a, b, s, a, b, s, x, y, s, a, b), Arrays.asList(l1, l1, l1, l1, l1, l1, l2, l2, l2, l1, l1));
		assertEquals(9.996534024760905, p2, 0.00000000000001);
	}

	@SuppressWarnings("serial")
	@Test
	public void test_calculatePerplexity_differentMaxOrders() {
		
		final CharIndexer charIndexer = new CharIndexer();
		charIndexer.index(new String[] { "a", "b", "x", "y", " " });
		
		final Indexer<String> langIndexer = new HashMapIndexer<String>();
		langIndexer.index(new String[]{ "Lang1", "Lang2" });
		
		final int a = charIndexer.getIndex("a");
		final int b = charIndexer.getIndex("b");
		final int x = charIndexer.getIndex("x");
		final int y = charIndexer.getIndex("y");
		final int s = charIndexer.getIndex(" ");
		
		final int l1 = langIndexer.getIndex("Lang1");
		final int l2 = langIndexer.getIndex("Lang2");
		
		final int[] ctx_ = new int[] {};
		final int[] ctx_a = new int[] { a };
		final int[] ctx_ab = new int[] { a , b };
		final int[] ctx_ab_ = new int[] { a , b , s };
		final int[] ctx_ab_a = new int[] { a, b , s , a };
		final int[] ctx_b_ab = new int[] { b, s, a , b };
		final int[] ctx_x = new int[] { x };
		final int[] ctx_xy = new int[] { x , y };
		
		final SingleLanguageModel lang1Lm = new SingleLanguageModel() {
			@Override public int getMaxOrder() { return 5; }
			@Override public double getCharNgramProb(int[] context, int c) {
				if (c == a) {
					if (sameIntArray(context, ctx_)) return 0.11; 
					if (sameIntArray(context, ctx_ab_)) return 0.12; 
				}
				if (c == b) {
					if (sameIntArray(context, ctx_a)) return 0.13; 
					if (sameIntArray(context, ctx_ab_a)) return 0.14; 
				}
				if (c == s) {
					if (sameIntArray(context, ctx_ab)) return 0.15; 
					if (sameIntArray(context, ctx_b_ab)) return 0.16; 
				}
				throw new RuntimeException("getCharNgramProb(" + intArrayToList(context) + ", " + c + ")");
			}
			@Override public Indexer<String> getCharacterIndexer() { return charIndexer; }
			@Override public Set<Integer> getActiveCharacters() { throw new RuntimeException(); }
			@Override public int[] shrinkContext(int[] originalContext) { throw new RuntimeException(); }
			@Override public boolean containsContext(int[] context) { throw new RuntimeException(); }
		};

		final SingleLanguageModel lang2Lm = new SingleLanguageModel() {
			@Override public int getMaxOrder() { return 4; }
			@Override public double getCharNgramProb(int[] context, int c) {
				if (c == x) {
					if (sameIntArray(context, ctx_)) return 0.21; 
				}
				if (c == y) {
					if (sameIntArray(context, ctx_x)) return 0.22; 
				}
				if (c == s) {
					if (sameIntArray(context, ctx_xy)) return 0.23; 
				}
				throw new RuntimeException("getCharNgramProb(" + intArrayToList(context) + ", " + c + ")");
			}
			@Override public Indexer<String> getCharacterIndexer() { return charIndexer; }
			@Override public Set<Integer> getActiveCharacters() { throw new RuntimeException(); }
			@Override public int[] shrinkContext(int[] originalContext) { throw new RuntimeException(); }
			@Override public boolean containsContext(int[] context) { throw new RuntimeException(); }
		};
		
		final CodeSwitchLanguageModel csLm = new CodeSwitchLanguageModel() {
			@Override public double getCharNgramProb(int[] context, int c) { throw new RuntimeException(); }
			@Override public Indexer<String> getCharacterIndexer() { return charIndexer; }
			@Override public Indexer<String> getLanguageIndexer() { return langIndexer; }
			@Override public SingleLanguageModel get(int language) { 
				if (language == langIndexer.getIndex("Lang1")) return lang1Lm;
				if (language == langIndexer.getIndex("Lang2")) return lang2Lm;
				throw new RuntimeException();
			}
			@Override public double languagePrior(int language)  { 
				if (language == langIndexer.getIndex("Lang1")) return 0.31;
				throw new RuntimeException();
			}
			@Override public double languageTransitionProb(int fromLanguage, int destLanguage) { 
				if (fromLanguage == langIndexer.getIndex("Lang1")) {
					if (destLanguage == langIndexer.getIndex("Lang1")) return 0.32;
					if (destLanguage == langIndexer.getIndex("Lang2")) return 0.33;
				}
				if (fromLanguage == langIndexer.getIndex("Lang2")) {
					if (destLanguage == langIndexer.getIndex("Lang1")) return 0.35;
					if (destLanguage == langIndexer.getIndex("Lang2")) return 0.34;
				}
				throw new RuntimeException();
			}
			@Override public double getProbKeepSameLanguage() { throw new RuntimeException(); }
		};

		
		LmPerplexity lmPerplexity = new LmPerplexity(csLm);
		
		/*
		 * Lang1: a,b
		 * Lang2: x,y
		 * 
		 * "ab ab xy ab"
		 * 
		 * P1(a|[])     * P(L1)        =  0.11 * 0.31
		 * P1(b|[a])    * P(L1|L1)     =  0.13 * 1.00
		 * P1( |[ab])   * P(L1|L1)     =  0.15 * 1.00
		 * P1(a|[ab ])  * P(L1|L1)     =  0.12 * 0.32
		 * P1(b|[ab a]) * P(L1|L1)     =  0.14 * 1.00
		 * P1( |[b ab]) * P(L1|L1)     =  0.16 * 1.00
		 * P2(x|[])     * P(L2|L1)     =  0.21 * 0.33
		 * P2(y|[x])    * P(L2|L2)     =  0.22 * 1.00
		 * P2( |[xy])   * P(L2|L2)     =  0.23 * 1.00
		 * P1(a|[])     * P(L1|L2)     =  0.11 * 0.35
		 * P1(b|[a])    * P(L1|L1)     =  0.13 * 1.00
		 *                               --------------------
		 *                            =  1.0038205132552398E-11 ^(-1/11)
		 *                            =  9.996534024760905
		 */
		double p2 = lmPerplexity.perplexity(Arrays.asList(a, b, s, a, b, s, x, y, s, a, b), Arrays.asList(l1, l1, l1, l1, l1, l1, l2, l2, l2, l1, l1));
		assertEquals(9.996534024760905, p2, 0.00000000000001);
	}

	private boolean sameIntArray(int[] a, int[] b) {
		if (a.length != b.length)
			return false;
		for (int i = 0; i < a.length; ++i) {
			if (a[i] != b[i])
				return false;
		}
		return true;
	}

}
