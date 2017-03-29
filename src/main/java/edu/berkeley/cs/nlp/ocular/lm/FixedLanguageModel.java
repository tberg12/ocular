package edu.berkeley.cs.nlp.ocular.lm;


import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Set;

import com.sun.org.apache.bcel.internal.generic.LMUL;

import edu.berkeley.cs.nlp.ocular.data.textreader.CharIndexer;
import edu.berkeley.cs.nlp.ocular.data.textreader.Charset;
import sun.security.util.Length;
import tberg.murphy.indexer.Indexer;

public class FixedLanguageModel {
	
	private Indexer<String> charIndexer;
	private int maxOrder;
	List<Integer> fixedText;
	
	private int[] counts;
	
	private double[][] transitionProb;
	
	public FixedLanguageModel(String fileName) {
		maxOrder = 1;
		charIndexer = new CharIndexer();		
		fixedText = new ArrayList<Integer>();
		
		double eps = 1e-15;		
		double fixedProb = 1-eps;
		
		charIndexer.getIndex(Charset.HYPHEN);
		
		try {
	      BufferedReader in = new BufferedReader(new InputStreamReader(new FileInputStream(fileName), "UTF-8"));
	      
	      while (in.ready()) {
	        String line = in.readLine();
	        for (int i = 0; i < line.length(); i++) {
	        	String toAdd = Character.toString(line.charAt(i));
	        	int index = charIndexer.getIndex(toAdd);
	        	fixedText.add(index);
	        }
	        fixedText.add(charIndexer.getIndex(Charset.SPACE));
	      }
	      in.close();
	    } catch (IOException e) {
	      throw new RuntimeException(e);
	    }

		charIndexer.getIndex(Charset.SPACE);
		
		double subProb = eps/(charIndexer.size()-1);
		System.out.println(subProb);
		
		this.transitionProb = new double[charIndexer.size()][charIndexer.size()];
		
		for (int i = 0; i<this.transitionProb.length; i++) {
			Arrays.fill(this.transitionProb[i], subProb);
			this.transitionProb[i][i] = fixedProb;
		}
		
		this.counts = new int[charIndexer.size()];
		Arrays.fill(counts, 0);
		
		for (int c : fixedText) {
			counts[c] += 1;
		}
		
	}
	

	public double getCharNgramProb(int pos, int c) {		
		return transitionProb[this.getCharAtPos(pos)][c];
	}
	
	public int getCharAtPos(int pos) {
		if (pos < fixedText.size())
			return fixedText.get(pos);
		return charIndexer.getIndex(Charset.SPACE);
	}
	
	public Indexer<String> getCharacterIndexer() {
		return charIndexer;
	}
	
	public Set<Integer> getActiveCharacters() {
		// TODO Auto-generated method stub
		return null;
	}
	

	public int getMaxOrder() {
		return maxOrder;
	}
	

	public int[] shrinkContext(int[] originalContext) {
		// TODO Auto-generated method stub
		return null;
	}
	

	public boolean containsContext(int[] context) {
		// TODO Auto-generated method stub
		return false;
	}
	
	public void updateProbs(String guess) {
		
		for (int i = 0; i<this.transitionProb.length; i++) {
			Arrays.fill(this.transitionProb[i], 0.0);
		}		
							
		for (int pos=0; pos<guess.length(); pos++) {
			transitionProb[getCharAtPos(pos)][getCharacterIndexer().getIndex(String.valueOf(guess.charAt(pos)))] += 1;
		}
		
		for (int i = 0; i<this.transitionProb.length; i++) {
			for (int j = 0; j<this.transitionProb[i].length; j++) {
				transitionProb[i][j] /= counts[i];
			}
		}		
	}
	
}