package edu.berkeley.cs.nlp.ocular.util;

import static edu.berkeley.cs.nlp.ocular.util.Tuple2.Tuple2;
import static java.util.Arrays.asList;
import static org.junit.Assert.*;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.junit.Test;

/**
 * @author Dan Garrette (dhgarrette@gmail.com)
 */
public class CollectionHelperTests {

	@Test
	public void test_map1() {
		Map<String, Integer> m1 = CollectionHelper.map1("two", 2);
		assertEquals(1, m1.size());
		assertEquals(Integer.valueOf(2), m1.get("two"));
		assertNull(m1.get("three"));
	}

	@Test
	public void test_makeMap() {
		Tuple2<String, Integer> t1 = Tuple2("one", 21);
		Tuple2<String, Integer> t2 = Tuple2("two", 22);
		Tuple2<String, Integer> t3 = Tuple2("three", 23);
		
		Map<String, Integer> m0 = CollectionHelper.makeMap();
		assertEquals(0, m0.size());
		assertNull(m0.get("four"));
		
		Map<String, Integer> m1 = CollectionHelper.makeMap(t1);
		assertEquals(1, m1.size());
		assertEquals(Integer.valueOf(21), m1.get("one"));
		assertNull(m1.get("four"));
		
		Map<String, Integer> m2 = CollectionHelper.makeMap(t1, t2);
		assertEquals(2, m2.size());
		assertEquals(Integer.valueOf(21), m2.get("one"));
		assertEquals(Integer.valueOf(22), m2.get("two"));
		assertNull(m2.get("four"));
		
		Map<String, Integer> m3 = CollectionHelper.makeMap(t1, t2, t3);
		assertEquals(3, m3.size());
		assertEquals(Integer.valueOf(21), m3.get("one"));
		assertEquals(Integer.valueOf(22), m3.get("two"));
		assertEquals(Integer.valueOf(23), m3.get("three"));
		assertNull(m3.get("four"));
	}

	@Test
	public void test_Map_getOrElse() {
		Tuple2<String, Integer> t1 = Tuple2("one", 21);
		Tuple2<String, Integer> t2 = Tuple2("two", 22);
		Map<String, Integer> m2 = CollectionHelper.makeMap(t1, t2);

		assertEquals(2, m2.size());
		assertEquals(Integer.valueOf(21), m2.get("one"));
		assertEquals(Integer.valueOf(22), m2.get("two"));
		assertNull(m2.get("four"));

		assertEquals(Integer.valueOf(21), CollectionHelper.getOrElse(m2, "one", Integer.valueOf(131)));
		assertEquals(Integer.valueOf(22), CollectionHelper.getOrElse(m2, "two", Integer.valueOf(132)));
		assertEquals(Integer.valueOf(134), CollectionHelper.getOrElse(m2, "four", Integer.valueOf(134)));
	}

	@Test
	public void test_makeSet_collection() {
		Set<String> m0 = CollectionHelper.makeSet(new ArrayList<String>());
		assertEquals(0, m0.size());
		assertFalse(m0.contains("four"));
		
		Set<String> m1 = CollectionHelper.makeSet(Arrays.asList("one"));
		assertEquals(1, m1.size());
		assertTrue(m1.contains("one"));
		assertFalse(m1.contains("four"));
		
		Set<String> m2 = CollectionHelper.makeSet(Arrays.asList("one", "two"));
		assertEquals(2, m2.size());
		assertTrue(m2.contains("one"));
		assertTrue(m2.contains("two"));
		assertFalse(m2.contains("four"));
		
		Set<String> m3 = CollectionHelper.makeSet(Arrays.asList("one", "two", "three"));
		assertEquals(3, m3.size());
		assertTrue(m3.contains("one"));
		assertTrue(m3.contains("two"));
		assertTrue(m3.contains("three"));
		assertFalse(m3.contains("four"));

		Set<String> m3b = CollectionHelper.makeSet(Arrays.asList("one", "two", "three", "two"));
		assertEquals(3, m3b.size());
		assertTrue(m3b.contains("one"));
		assertTrue(m3b.contains("two"));
		assertTrue(m3b.contains("three"));
		assertFalse(m3b.contains("four"));
	}

	@Test
	public void test_makeSet_varargs() {
		Set<String> m0 = CollectionHelper.makeSet();
		assertEquals(0, m0.size());
		assertFalse(m0.contains("four"));
		
		Set<String> m1 = CollectionHelper.makeSet("one");
		assertEquals(1, m1.size());
		assertTrue(m1.contains("one"));
		assertFalse(m1.contains("four"));
		
		Set<String> m2 = CollectionHelper.makeSet("one", "two");
		assertEquals(2, m2.size());
		assertTrue(m2.contains("one"));
		assertTrue(m2.contains("two"));
		assertFalse(m2.contains("four"));
		
		Set<String> m3 = CollectionHelper.makeSet("one", "two", "three");
		assertEquals(3, m3.size());
		assertTrue(m3.contains("one"));
		assertTrue(m3.contains("two"));
		assertTrue(m3.contains("three"));
		assertFalse(m3.contains("four"));

		Set<String> m3b = CollectionHelper.makeSet("one", "two", "three", "two");
		assertEquals(3, m3b.size());
		assertTrue(m3b.contains("one"));
		assertTrue(m3b.contains("two"));
		assertTrue(m3b.contains("three"));
		assertFalse(m3b.contains("four"));
	}

	@Test
	public void test_setUnion() {
		Set<String> s1 = CollectionHelper.makeSet("one", "two", "three");
		Set<String> s2 = CollectionHelper.makeSet("two", "three", "four");
		Set<String> su = CollectionHelper.setUnion(s1, s2);
		assertEquals(4, su.size());
		assertTrue(su.contains("one"));
		assertTrue(su.contains("two"));
		assertTrue(su.contains("three"));
		assertTrue(su.contains("four"));
		assertFalse(su.contains("five"));
	}

	@Test
	public void test_setIntersection() {
		Set<String> s1 = CollectionHelper.makeSet("one", "two", "three");
		Set<String> s2 = CollectionHelper.makeSet("two", "three", "four");
		Set<String> su = CollectionHelper.setIntersection(s1, s2);
		assertEquals(2, su.size());
		assertFalse(su.contains("one"));
		assertTrue(su.contains("two"));
		assertTrue(su.contains("three"));
		assertFalse(su.contains("four"));
		assertFalse(su.contains("five"));
	}

	@Test
	public void test_setDiff() {
		Set<String> s1 = CollectionHelper.makeSet("zero", "one", "two", "three");
		Set<String> s2 = CollectionHelper.makeSet("two", "three", "four");
		Set<String> su = CollectionHelper.setDiff(s1, s2);
		assertEquals(2, su.size());
		assertTrue(su.contains("zero"));
		assertTrue(su.contains("one"));
		assertFalse(su.contains("two"));
		assertFalse(su.contains("three"));
		assertFalse(su.contains("four"));
		assertFalse(su.contains("five"));
	}

	@Test
	public void test_makeList_collection() {
		List<String> l1 = CollectionHelper.makeList(CollectionHelper.makeSet("one", "two", "three"));
		assertEquals(3, l1.size());
		assertTrue(l1.contains("one"));
		assertTrue(l1.contains("two"));
		assertTrue(l1.contains("three"));
		assertFalse(l1.contains("four"));
	}

	@Test
	public void test_makeList_varargs() {
		List<String> l1 = CollectionHelper.makeList("one", "two", "three");
		assertEquals(3, l1.size());
		assertEquals("one", l1.get(0));
		assertEquals("two", l1.get(1));
		assertEquals("three", l1.get(2));
	}

	@Test
	public void test_fillList() {
		List<String> l1 = CollectionHelper.fillList(3, "one");
		assertEquals(3, l1.size());
		assertEquals("one", l1.get(0));
		assertEquals("one", l1.get(1));
		assertEquals("one", l1.get(2));
	}
	
	@Test
	public void test_listCat() {
		List<String> l0 = CollectionHelper.listCat();
		assertEquals(0, l0.size());
		
		List<String> l1 = CollectionHelper.listCat(Arrays.asList("one", "two", "three"));
		assertEquals(3, l1.size());
		assertEquals("one", l1.get(0));
		assertEquals("two", l1.get(1));
		assertEquals("three", l1.get(2));
		
		List<String> l2 = CollectionHelper.listCat(Arrays.asList("one", "two", "three"), Arrays.<String>asList(), Arrays.asList("four", "five"));
		assertEquals(5, l2.size());
		assertEquals("one", l2.get(0));
		assertEquals("two", l2.get(1));
		assertEquals("three", l2.get(2));
		assertEquals("four", l2.get(3));
		assertEquals("five", l2.get(4));
	}

	@Test
	public void test_sliding() {
		Iterator<List<String>> s1 = CollectionHelper.sliding(Arrays.asList("one", "two", "three", "four", "five"), 3);
		assertTrue(s1.hasNext());
		List<String> s10 = s1.next();
		assertEquals(3, s10.size());
		assertEquals("one", s10.get(0));
		assertEquals("two", s10.get(1));
		assertEquals("three", s10.get(2));
		assertTrue(s1.hasNext());
		List<String> s11 = s1.next();
		assertEquals(3, s11.size());
		assertEquals("two", s11.get(0));
		assertEquals("three", s11.get(1));
		assertEquals("four", s11.get(2));
		assertTrue(s1.hasNext());
		List<String> s12 = s1.next();
		assertEquals(3, s12.size());
		assertEquals("three", s12.get(0));
		assertEquals("four", s12.get(1));
		assertEquals("five", s12.get(2));
		assertFalse(s1.hasNext());

		Iterator<List<String>> s2 = CollectionHelper.sliding(Arrays.asList("one", "two"), 3);
		assertFalse(s2.hasNext());
	}
	
	@Test
	public void test_List_take() {
		{
		List<String> l1 = CollectionHelper.take(Arrays.asList("one", "two", "three", "four", "five"), 3);
		assertEquals(3, l1.size());
		assertEquals("one", l1.get(0));
		assertEquals("two", l1.get(1));
		assertEquals("three", l1.get(2));
		}
		{
		List<String> l1 = CollectionHelper.take(Arrays.asList("one", "two", "three", "four", "five"), 5);
		assertEquals(5, l1.size());
		assertEquals("one", l1.get(0));
		assertEquals("two", l1.get(1));
		assertEquals("three", l1.get(2));
		assertEquals("four", l1.get(3));
		assertEquals("five", l1.get(4));
		}
		{
		List<String> l1 = CollectionHelper.take(Arrays.asList("one", "two", "three", "four", "five"), 7);
		assertEquals(5, l1.size());
		assertEquals("one", l1.get(0));
		assertEquals("two", l1.get(1));
		assertEquals("three", l1.get(2));
		assertEquals("four", l1.get(3));
		assertEquals("five", l1.get(4));
		}
		{
		List<String> l1 = CollectionHelper.take(Arrays.asList("one", "two", "three", "four", "five"), 0);
		assertEquals(0, l1.size());
		}
	}

	@Test
	public void test_List_takeRight() {
		{
		List<String> l1 = CollectionHelper.takeRight(Arrays.asList("one", "two", "three", "four", "five"), 3);
		assertEquals(3, l1.size());
		assertEquals("three", l1.get(0));
		assertEquals("four", l1.get(1));
		assertEquals("five", l1.get(2));
		}
		{
		List<String> l1 = CollectionHelper.takeRight(Arrays.asList("one", "two", "three", "four", "five"), 5);
		assertEquals(5, l1.size());
		assertEquals("one", l1.get(0));
		assertEquals("two", l1.get(1));
		assertEquals("three", l1.get(2));
		assertEquals("four", l1.get(3));
		assertEquals("five", l1.get(4));
		}
		{
		List<String> l1 = CollectionHelper.takeRight(Arrays.asList("one", "two", "three", "four", "five"), 7);
		assertEquals(5, l1.size());
		assertEquals("one", l1.get(0));
		assertEquals("two", l1.get(1));
		assertEquals("three", l1.get(2));
		assertEquals("four", l1.get(3));
		assertEquals("five", l1.get(4));
		}
		{
		List<String> l1 = CollectionHelper.takeRight(Arrays.asList("one", "two", "three", "four", "five"), 0);
		assertEquals(0, l1.size());
		}
	}

	@Test
	public void test_List_drop() {
		{
		List<String> l1 = CollectionHelper.drop(Arrays.asList("one", "two", "three", "four", "five"), 2);
		assertEquals(3, l1.size());
		assertEquals("three", l1.get(0));
		assertEquals("four", l1.get(1));
		assertEquals("five", l1.get(2));
		}
		{
		List<String> l1 = CollectionHelper.drop(Arrays.asList("one", "two", "three", "four", "five"), 5);
		assertEquals(0, l1.size());
		}
		{
		List<String> l1 = CollectionHelper.drop(Arrays.asList("one", "two", "three", "four", "five"), 7);
		assertEquals(0, l1.size());
		}
		{
		List<String> l1 = CollectionHelper.drop(Arrays.asList("one", "two", "three", "four", "five"), 0);
		assertEquals(5, l1.size());
		assertEquals("one", l1.get(0));
		assertEquals("two", l1.get(1));
		assertEquals("three", l1.get(2));
		assertEquals("four", l1.get(3));
		assertEquals("five", l1.get(4));
		}
	}

	@Test
	public void test_intArrayToList() {
		int[] a1 = { 4, 5, 6, 7 };
		List<Integer> l1 = CollectionHelper.intArrayToList(a1);
		assertEquals(4, l1.size());
		assertEquals(Integer.valueOf(4), l1.get(0));
		assertEquals(Integer.valueOf(5), l1.get(1));
		assertEquals(Integer.valueOf(6), l1.get(2));
		assertEquals(Integer.valueOf(7), l1.get(3));
	}

	@Test
	public void test_intListToArray() {
		int[] a1a = CollectionHelper.intListToArray(Arrays.asList(4, 5, 6, 7));
		int[] a1b = { 4, 5, 6, 7 };
		assertArrayEquals(a1b, a1a);
	}
	
	@Test
	public void test_longestCommonPrefix() {
		assertEquals(0, CollectionHelper.longestCommonPrefix(asList(Arrays.<Integer>asList(), Arrays.<Integer>asList())));
		assertEquals(0, CollectionHelper.longestCommonPrefix(asList(Arrays.<Integer>asList(1), Arrays.<Integer>asList())));
		assertEquals(0, CollectionHelper.longestCommonPrefix(asList(Arrays.<Integer>asList(), Arrays.<Integer>asList(2))));
		assertEquals(0, CollectionHelper.longestCommonPrefix(asList(Arrays.<Integer>asList(1), Arrays.<Integer>asList(2))));
		assertEquals(2, CollectionHelper.longestCommonPrefix(asList(Arrays.<Integer>asList(1, 2), Arrays.<Integer>asList(1, 2, 3, 4))));
		assertEquals(2, CollectionHelper.longestCommonPrefix(asList(Arrays.<Integer>asList(1, 2, 3, 4), Arrays.<Integer>asList(1, 2))));
		assertEquals(3, CollectionHelper.longestCommonPrefix(asList(Arrays.<Integer>asList(1, 2, 3), Arrays.<Integer>asList(1, 2, 3))));
		assertEquals(2, CollectionHelper.longestCommonPrefix(asList(Arrays.<Integer>asList(1, 2, 3), Arrays.<Integer>asList(1, 2), Arrays.<Integer>asList(1, 2, 3, 4))));
	}
}
