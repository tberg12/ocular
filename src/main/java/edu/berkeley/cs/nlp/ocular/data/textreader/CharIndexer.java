package edu.berkeley.cs.nlp.ocular.data.textreader;

import java.util.Collection;

import tberg.murphy.indexer.HashMapIndexer;
import tberg.murphy.indexer.Indexer;

/**
 * @author Dan Garrette (dhgarrette@gmail.com)
 */
public class CharIndexer implements Indexer<String> {
	private static final long serialVersionUID = 3212987272223100239L;
	
	private Indexer<String> delegate;
	
	public CharIndexer() {
		delegate = new HashMapIndexer<String>();
	}

	public boolean contains(String object) {
		return delegate.contains(Charset.normalizeChar(object)); 
	}
	
	public int getIndex(String object) { 
		return delegate.getIndex(Charset.normalizeChar(object)); 
	}
	
	public void index(String[] vect) { 
		for (String x : vect)
			getIndex(x);
	}
	
	public boolean locked() { return delegate.locked(); }
	public void lock() { delegate.lock(); }
	public int size() { return delegate.size(); }
	public String getObject(int index) { return delegate.getObject(index); }
	public void forgetIndexLookup() { delegate.forgetIndexLookup(); }
	public Collection<String> getObjects() { return delegate.getObjects(); }

}
