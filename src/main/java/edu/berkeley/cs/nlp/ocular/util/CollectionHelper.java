package edu.berkeley.cs.nlp.ocular.util;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;

import tuple.Pair;

public class CollectionHelper {

	public static <K, V> Map<K, V> map1(K key, V value) {
		return Collections.singletonMap(key, value);
	}

	public static <K, V> Map<K, V> makeMap(Pair<K, V>... pairs) {
		if (pairs.length == 0) {
			return Collections.<K, V> emptyMap();
		}
		else if (pairs.length == 1) {
			Pair<K, V> p = pairs[0];
			return Collections.singletonMap(p.getFirst(), p.getSecond());
		}
		else {
			Map<K, V> m = new HashMap<K, V>();
			for (Pair<K, V> pair : pairs) {
				m.put(pair.getFirst(), pair.getSecond());
			}
			return m;
		}
	}

	public static <K, V> V getOrElse(Map<K, V> m, K k, V def) {
		V v = m.get(k);
		if (v != null)
			return v;
		else
			return def;
	}

	public static <A> Set<A> makeSet(A... xs) {
		if (xs.length == 0)
			return Collections.<A> emptySet();
		else if (xs.length == 1)
			return Collections.singleton(xs[0]);
		else {
			Set<A> set = new HashSet<A>(xs.length);
			Collections.addAll(set, xs);
			return set;
		}
	}

	public static <A> Set<A> setUnion(Set<A>... sets) {
		if (sets.length == 0)
			return Collections.<A> emptySet();
		else if (sets.length == 1)
			return sets[0];
		else {
			Set<A> set = new HashSet<A>();
			for (Set<A> xs : sets)
				set.addAll(xs);
			return set;
		}
	}

	public static <A> Set<A> setDiff(Set<A> a, Set<A> b) {
		Set<A> set = new HashSet<A>();
		for (A x : a)
			if (!b.contains(x)) 
				set.add(x);
		return set;
	}

	public static <A> Set<A> setIntersection(Set<A> a, Set<A> b) {
		Set<A> set = new HashSet<A>();
		for (A x : a)
			if (b.contains(x)) 
				set.add(x);
		return set;
	}

	public static <A> List<A> listCat(List<A>... lists) {
		if (lists.length == 0)
			return Collections.<A> emptyList();
		else if (lists.length == 1)
			return lists[0];
		else {
			List<A> full = new ArrayList<A>();
			for (List<A> xs : lists)
				full.addAll(xs);
			return full;
		}
	}

	public static <A> Iterator<List<A>> sliding(List<A> list, int n) {
		return new SlidingIterator<A>(list, n);
	}

	private static class SlidingIterator<A> implements Iterator<List<A>> {
		private List<A> list;
		private int n;
		private int position;

		public SlidingIterator(List<A> list, int n) {
			if (n <= 0)
				throw new IllegalArgumentException("`n` must be greater than zero");
			this.list = list;
			this.n = n;
			this.position = 0;
		}

		public boolean hasNext() {
			return position < list.size() - n + 1;
		}

		public List<A> next() {
			List<A> result = new ArrayList<A>();
			for (int j = 0; j < n; ++j)
				result.add(list.get(position + j));
			++position;
			return result;
		}

		public void remove() {
			throw new RuntimeException("remove not supported on SlidingIterator");
		}
	}

	public static <A> List<A> take(List<A> list, int n) {
		List<A> result = new ArrayList<A>();
		for (int j = 0; j < Math.min(n, list.size()); ++j)
			result.add(list.get(j));
		return result;
	}

	public static <A> List<A> takeRight(List<A> list, int n) {
		List<A> result = new ArrayList<A>();
		for (int j = list.size() - n; j < list.size(); ++j)
			result.add(list.get(j));
		return result;
	}

	public static List<Integer> intArrayToList(int[] a) {
		List<Integer> l = new ArrayList<Integer>(a.length);
		for (int x: a) l.add(x);
		return l;
	}
	
}
