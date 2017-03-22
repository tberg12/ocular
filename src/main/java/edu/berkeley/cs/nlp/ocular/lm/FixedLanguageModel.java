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
	private final double fixedProb;
	private final double subProb;
	
	public FixedLanguageModel(String fileName) {
		maxOrder = 1;
		charIndexer = new CharIndexer();		
		fixedText = new ArrayList<Integer>();
		
		fixedProb = 1-1e-15;
		
		charIndexer.getIndex(Charset.SPACE);
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
		
		subProb = (1-fixedProb)/charIndexer.size();
		System.out.println(subProb);
	}
	

	public double getCharNgramProb(int pos, int c) {
		if (c == this.getCharAtPos(pos)) {
			return fixedProb;
		}
		
		return subProb;
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
	
}