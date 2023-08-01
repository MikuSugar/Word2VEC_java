//
// Source code recreated from a .class file by IntelliJ IDEA
// (powered by Fernflower decompiler)
//

package com.ansj.vec.util;

import java.util.HashMap;
import java.util.Iterator;
import java.util.Map.Entry;

public class MapCount<T>
{
    private HashMap<T, Integer> hm = null;

    public MapCount()
    {
        this.hm = new HashMap<>();
    }

    public MapCount(int initialCapacity)
    {
        this.hm = new HashMap<>(initialCapacity);
    }

    /**
     * Adds the given value to the existing value associated with the specified key in the hashmap.
     *
     * @param t the key to which the value is to be added
     * @param n the value to be added
     */
    public void add(T t, int n)
    {
        this.hm.merge(t, n, Integer::sum);
    }

    public void add(T t)
    {
        this.add(t, 1);
    }

    public int size()
    {
        return this.hm.size();
    }

    public void remove(T t)
    {
        this.hm.remove(t);
    }

    public HashMap<T, Integer> get()
    {
        return this.hm;
    }

    public String getDic()
    {
        Iterator<Entry<T, Integer>> iterator = this.hm.entrySet().iterator();
        StringBuilder sb = new StringBuilder();
        Entry<T, Integer> next;

        while (iterator.hasNext())
        {
            next = iterator.next();
            sb.append(next.getKey());
            sb.append("\t");
            sb.append(next.getValue());
            sb.append("\n");
        }
        return sb.toString();
    }

}
