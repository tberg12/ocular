package edu.berkeley.cs.nlp.ocular.model;
//package com.halperta.ocr.ocular.model;
//
//import tberg.murphy.indexer.HashMapIndexer;
//import tberg.murphy.indexer.Indexer;
//
//import java.util.Arrays;
//import java.util.Collection;
//import java.util.HashMap;
//import java.util.HashSet;
//import java.util.Map;
//import java.util.Set;
//
//import edu.berkeley.cs.nlp.ocular.lm.BasicCodeSwitchLanguageModel;
//import edu.berkeley.cs.nlp.ocular.lm.CodeSwitchLanguageModel;
//import edu.berkeley.cs.nlp.ocular.lm.SingleLanguageModel;
//import edu.berkeley.cs.nlp.ocular.model.SparseTransitionModel.TransitionState;
//
//import org.junit.Test;
//
//import tberg.murphy.tuple.Pair;
//
//import com.google.common.collect.Sets;
//
//import edu.berkeley.cs.nlp.ocular.model.TransitionStateType;
//import edu.berkeley.cs.nlp.ocular.util.Tuple3;
//import static edu.berkeley.cs.nlp.ocular.model.TransitionStateType.*;
//
//import com.halperta.ocr.ocular.model.PostViterbi.CharacterAlternatives;
//
///**
// * @author Dan Garrette (dhgarrette@gmail.com)
// */
//public class PostViterbiTests {
//
//	public class BasicLanguageCharState implements TransitionState {
//
//		public final int charIndex;
//		public final String language;
//		public final TransitionStateType type;
//
//		public BasicLanguageCharState(int charIndex, String language, TransitionStateType type) {
//			this.charIndex = charIndex;
//			this.language = language;
//			this.type = type;
//		}
//
//		public BasicLanguageCharState(int charIndex, String language) {
//			this(charIndex, language, TransitionStateType.TMPL);
//		}
//
//		public int getCharIndex() {
//			return charIndex;
//		}
//
//		public String getLanguage() {
//			return language;
//		}
//
//		public TransitionStateType getType() {
//			return type;
//		}
//
//		public String toString() {
//			return "(" + language + "," + type + "," + charIndex + ")";
//		}
//
//		public int getOffset() {
//			return -1;
//		}
//
//		public int getExposure() {
//			return -1;
//		}
//
//		public Collection<Pair<TransitionState, Double>> forwardTransitions() {
//			return null;
//		}
//
//		public Collection<Pair<TransitionState, Double>> nextLineStartStates() {
//			return null;
//		}
//
//		public double endLogProb() {
//			return -1;
//		}
//	}
//
//	@Test
//	public void test_run_1() {
//
//		final int maxOrder = 4;
//
//		final Indexer<String> charIndexer = new CharIndexer<String>();
//		charIndexer.getIndex(" ");
//		charIndexer.getIndex("a");
//		charIndexer.getIndex("b");
//		charIndexer.getIndex("c");
//		charIndexer.getIndex("d");
//		charIndexer.getIndex("e");
//		charIndexer.getIndex("f");
//		charIndexer.getIndex("g");
//		charIndexer.getIndex("h");
//		charIndexer.getIndex("i");
//		charIndexer.getIndex("-");
//		charIndexer.lock();
//
//		Map<String, Tuple3<SingleLanguageModel, Set<String>, Double>> subModelsAndPriors = new HashMap<String, Tuple3<SingleLanguageModel, Set<String>, Double>>();
//		SingleLanguageModel lmA = new SingleLanguageModel() {
//			public Set<Integer> getActiveCharacters() {
//				return Sets.newHashSet(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10);
//			}
//
//			public boolean containsContext(int[] context) {
//				if (Arrays.equals(context, new int[] { 1, 3, 6 })) return false;
//				if (Arrays.equals(context, new int[] { 1, 3, 7 })) return false;
//				if (Arrays.equals(context, new int[] { 3, 6 })) return false;
//				return true;
//			}
//
//			public double getCharNgramProb(int[] context, int c) {
//				return 0.1;
//			}
//
//			public Indexer<String> getCharacterIndexer() {
//				return charIndexer;
//			}
//
//			public int getMaxOrder() {
//				return maxOrder;
//			}
//			
//			public double logPerplexity(int[] chars) {
//				return -1.0;
//			}
//		};
//		Tuple3<SingleLanguageModel, Set<String>, Double> t = new Tuple3<SingleLanguageModel, Set<String>, Double>(lmA, new HashSet<String>(), 0.8);
//		subModelsAndPriors.put("A", t);
//		final CodeSwitchLanguageModel lm = new BasicCodeSwitchLanguageModel(subModelsAndPriors, charIndexer, 0.99, maxOrder);
//		int decodeBatchSize = 100;
//
//		CharacterAlternatives ambiguousDecoder = new CharacterAlternatives() {
//			public Set<String> get(String c) {
//				int i = charIndexer.getIndex(c);
//				if (i == 0) return Sets.newHashSet(" ");
//				if (i == 1 || i==2) return Sets.newHashSet("a", "b");
//				if (i == 3) return Sets.newHashSet("c");
//				if (i == 4||i==5) return Sets.newHashSet("d", "e");
//				if (i == 6 || i == 7) return Sets.newHashSet("f", "g");
//				if (i == 8) return Sets.newHashSet("h");
//				if (i == 9) return Sets.newHashSet("i");
//				if (i == 10) return Sets.newHashSet("-");
//				throw new RuntimeException("CharacterAlternatives.get(" + i + ")  not found");
//			}
//		};
//
//		PostViterbi pv = new PostViterbi(lm, ambiguousDecoder, false, decodeBatchSize, true);
//
//		//		0
//		//		a
//		//		b	
//		pv.run(new TransitionState[][] { new TransitionState[] { new BasicLanguageCharState(1, "A") } });
//
//		//		0	1
//		//		a	c
//		//		b		
//		pv.run(new TransitionState[][] { new TransitionState[] { new BasicLanguageCharState(1, "A"), new BasicLanguageCharState(3, "A") } });
//
//		//		0	1	2
//		//		a	c	d
//		//		b		e	
//		pv.run(new TransitionState[][] { new TransitionState[] { new BasicLanguageCharState(1, "A"), new BasicLanguageCharState(3, "A"), new BasicLanguageCharState(4, "A") } });
//
//		//		0	1	2	3
//		//		a	c	d	f
//		//		b		e	g	
//		pv.run(new TransitionState[][] { new TransitionState[] { new BasicLanguageCharState(1, "A"), new BasicLanguageCharState(3, "A"), new BasicLanguageCharState(4, "A"), new BasicLanguageCharState(6, "A") } });
//
//		//		0	1	2	3	4	5
//		//		a	c	d	f	h   i
//		//		b		e	g	
//		pv.run(new TransitionState[][] { new TransitionState[] { new BasicLanguageCharState(1, "A"), new BasicLanguageCharState(3, "A"), new BasicLanguageCharState(4, "A"), new BasicLanguageCharState(6, "A"), new BasicLanguageCharState(8, "A") } });
//
//		//		0	1	2	3	4	5
//		//		a	c	d	f	h   i
//		//		b		e	g	
//		pv.run(new TransitionState[][] { new TransitionState[] { new BasicLanguageCharState(1, "A"), new BasicLanguageCharState(3, "A"), new BasicLanguageCharState(4, "A"), new BasicLanguageCharState(6, "A"), new BasicLanguageCharState(8, "A"), new BasicLanguageCharState(9, "A") } });
//
//		//		0	1	2	3	|	4	5
//		//		a	c	d	f	|	h   i
//		//		b		e	g	|
//		pv.run(new TransitionState[][] { new TransitionState[] { new BasicLanguageCharState(1, "A"), new BasicLanguageCharState(3, "A"), new BasicLanguageCharState(4, "A"), new BasicLanguageCharState(6, "A") }, new TransitionState[] { new BasicLanguageCharState(8, "A"), new BasicLanguageCharState(9, "A") } });
//
//		//		0	1	2	3	|	5	6	7	8	9
//		//		a	c	d	f	|	_	h   a	d	i
//		//		b		e	g	|			b	e
//		pv.run(new TransitionState[][] { new TransitionState[] { new BasicLanguageCharState(1, "A"), new BasicLanguageCharState(3, "A"), new BasicLanguageCharState(4, "A"), new BasicLanguageCharState(6, "A") }, new TransitionState[] { new BasicLanguageCharState(0, "A"), new BasicLanguageCharState(8, "A"), new BasicLanguageCharState(1, "A"), new BasicLanguageCharState(4, "A"), new BasicLanguageCharState(9, "A") } });
//
//		//		0	1	2	3	4	|	6	7	8	9
//		//		a	c	d	f	_	|	h	a	d	i
//		//		b		e	g		|		b	e
//		pv.run(new TransitionState[][] { new TransitionState[] { new BasicLanguageCharState(1, "A"), new BasicLanguageCharState(3, "A"), new BasicLanguageCharState(4, "A"), new BasicLanguageCharState(6, "A") }, new TransitionState[] { new BasicLanguageCharState(0, "A"), new BasicLanguageCharState(8, "A"), new BasicLanguageCharState(1, "A"), new BasicLanguageCharState(4, "A"), new BasicLanguageCharState(9, "A") } });
//
//		//		0	1	2	3	x	|	5	6	7	8
//		//		a	c	d	f	-	|	h   a	d	i
//		//		b		e	g		|		b	e
//		pv.run(new TransitionState[][] { new TransitionState[] { new BasicLanguageCharState(1, "A"), new BasicLanguageCharState(3, "A"), new BasicLanguageCharState(4, "A"), new BasicLanguageCharState(6, "A"), new BasicLanguageCharState(10, "A", RMRGN_HPHN_INIT), new BasicLanguageCharState(0, "A", RMRGN_HPHN) }, new TransitionState[] { new BasicLanguageCharState(8, "A"), new BasicLanguageCharState(1, "A"), new BasicLanguageCharState(4, "A"), new BasicLanguageCharState(9, "A") } });
//
//		//		0	1	2	3	x	|	5	6	7	8
//		//		a	c	f	f	-	|	h   a	d	i
//		//		b		g	g		|		b	e
//		pv.run(new TransitionState[][] { new TransitionState[] { new BasicLanguageCharState(1, "A"), new BasicLanguageCharState(3, "A"), new BasicLanguageCharState(6, "A"), new BasicLanguageCharState(6, "A"), new BasicLanguageCharState(10, "A") }, new TransitionState[] { new BasicLanguageCharState(8, "A"), new BasicLanguageCharState(1, "A"), new BasicLanguageCharState(4, "A"), new BasicLanguageCharState(9, "A") } });
//
//	}
//}
