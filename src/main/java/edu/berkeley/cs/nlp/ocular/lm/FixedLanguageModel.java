package edu.berkeley.cs.nlp.ocular.lm;


import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Set;

import edu.berkeley.cs.nlp.ocular.data.textreader.CharIndexer;
import edu.berkeley.cs.nlp.ocular.data.textreader.Charset;
import tberg.murphy.indexer.Indexer;

public class FixedLanguageModel {
	
	private Indexer<String> charIndexer;
	private int maxOrder;
	List<Integer> fixedText;
	
	public FixedLanguageModel(String fileName) {
		maxOrder = 1;
		charIndexer = new CharIndexer();		
		fixedText = new ArrayList<Integer>();
		
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
	}
	

	public double getCharNgramProb() {
		// TODO Auto-generated method stub
		return 1;
	}
	
	public int getCharAtPos(int pos) {
		return fixedText.get(pos);
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