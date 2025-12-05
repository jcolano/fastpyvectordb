#!/usr/bin/env python3
"""
================================================================================
NEWS INTELLIGENCE QUERY INTERFACE
================================================================================

A user-friendly interface to query the news intelligence database.
Run this AFTER running news_intelligence_demo.py to populate the database.

Usage:
    python query_news_db.py                    # Interactive mode
    python query_news_db.py --search "AI"      # Quick search
    python query_news_db.py --entity "Apple"   # Entity lookup
    python query_news_db.py --topic "climate"  # Topic search

================================================================================
"""

import sys
import os
import numpy as np
import argparse
from datetime import datetime
from collections import defaultdict

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vectordb_optimized import VectorDB, Filter

# Optional imports
try:
    from graph import GraphDB
    HAS_GRAPH = True
except ImportError:
    HAS_GRAPH = False


class NewsQueryInterface:
    """
    User-friendly interface to query the news intelligence database.

    This is what an end-user would use to search and explore the news data.
    """

    def __init__(self, db_path: str = "./news_intelligence_db"):
        """Load the existing database."""
        print(f"Loading database from {db_path}...")

        self.db_path = db_path
        self.db = VectorDB(db_path)
        self.collection = self.db.get_collection("articles")

        # Load graph if available
        self.graph = None
        if HAS_GRAPH:
            graph_path = os.path.join(db_path, "graph")
            graph_file = os.path.join(graph_path, "graph.json")
            if os.path.exists(graph_file):
                self.graph = GraphDB(graph_path)
                print(f"  Graph loaded: {self.graph.node_count():,} nodes, {self.graph.edge_count():,} edges")
            else:
                print(f"  Graph not found at {graph_path}")
                print("  (Run news_intelligence_demo.py first to build the graph)")
        else:
            print("  Graph module not available")

        # Build topic embedding cache (simplified - in production use real embeddings)
        self._build_topic_embeddings()

        print(f"  Articles loaded: {self.collection.count():,}")
        print("Ready for queries!\n")

    def _build_topic_embeddings(self):
        """Build simple topic embeddings for search."""
        self.embedding_dim = 384
        self.topic_embeddings = {}

        # Create reproducible embeddings for common topics
        topics = [
            "artificial intelligence", "machine learning", "climate change",
            "stock market", "cryptocurrency", "space exploration", "healthcare",
            "politics", "technology", "business", "science", "sports"
        ]

        np.random.seed(42)
        for topic in topics:
            emb = np.random.randn(self.embedding_dim).astype(np.float32)
            emb = emb / np.linalg.norm(emb)
            self.topic_embeddings[topic] = emb

    def _get_query_embedding(self, query: str) -> np.ndarray:
        """Get or create embedding for a query."""
        query_lower = query.lower()

        # Check if we have a cached embedding
        if query_lower in self.topic_embeddings:
            return self.topic_embeddings[query_lower]

        # Create a deterministic embedding based on query
        # In production, you'd use a real embedding model here
        seed = hash(query_lower) % (2**32)
        rng = np.random.default_rng(seed)
        emb = rng.standard_normal(self.embedding_dim).astype(np.float32)
        emb = emb / np.linalg.norm(emb)
        return emb

    # =========================================================================
    # SEARCH METHODS - What users actually call
    # =========================================================================

    def search(self, query: str, k: int = 10,
               category: str = None,
               source: str = None,
               sentiment: str = None,
               year: int = None) -> list:
        """
        Search for articles matching a query.

        Args:
            query: Search query (e.g., "artificial intelligence", "climate change")
            k: Number of results to return
            category: Filter by category (technology, business, politics, etc.)
            source: Filter by source (TechCrunch, Bloomberg, etc.)
            sentiment: Filter by sentiment (positive, negative, neutral)
            year: Filter by publication year

        Returns:
            List of matching articles with scores
        """
        # Get query embedding
        query_embedding = self._get_query_embedding(query)

        # Build filter
        filter_obj = None
        filter_conditions = []

        if category:
            filter_conditions.append(Filter.eq("category", category))
        if source:
            filter_conditions.append(Filter.eq("source", source))
        if sentiment:
            filter_conditions.append(Filter.eq("sentiment", sentiment))
        if year:
            filter_conditions.append(Filter.eq("published_year", year))

        if len(filter_conditions) == 1:
            filter_obj = filter_conditions[0]
        elif len(filter_conditions) > 1:
            filter_obj = Filter.and_(filter_conditions)

        # Execute search
        results = self.collection.search(query_embedding, k=k, filter=filter_obj)

        return results

    def search_by_category(self, category: str, query: str = None, k: int = 10) -> list:
        """Search within a specific category."""
        if query:
            return self.search(query, k=k, category=category)
        else:
            # Just return random articles from category
            query_embedding = np.random.randn(self.embedding_dim).astype(np.float32)
            return self.collection.search(
                query_embedding, k=k,
                filter=Filter.eq("category", category)
            )

    def search_by_sentiment(self, sentiment: str, query: str = None, k: int = 10) -> list:
        """Search for articles with specific sentiment."""
        if query:
            return self.search(query, k=k, sentiment=sentiment)
        else:
            query_embedding = np.random.randn(self.embedding_dim).astype(np.float32)
            return self.collection.search(
                query_embedding, k=k,
                filter=Filter.eq("sentiment", sentiment)
            )

    def get_article(self, article_id: str) -> dict:
        """Get a specific article by ID."""
        result = self.collection.get(article_id, include_vector=False)
        return result

    def read_article(self, article_id: str) -> dict:
        """
        Read a full article by ID.

        Args:
            article_id: The article ID (e.g., "article_00000001")

        Returns:
            Dictionary with full article details
        """
        # Get from vector DB
        result = self.collection.get(article_id, include_vector=False)

        if not result:
            return {"error": f"Article '{article_id}' not found"}

        # The get() method returns {"id": ..., "metadata": {...}}
        metadata = result.get("metadata", {})

        article = {
            "id": article_id,
            "headline": metadata.get("headline", "N/A"),
            "content": metadata.get("content", "N/A"),
            "category": metadata.get("category", "N/A"),
            "topic": metadata.get("topic", "N/A"),
            "source": metadata.get("source", "N/A"),
            "published_date": metadata.get("published_date", "N/A"),
            "sentiment": metadata.get("sentiment", "N/A"),
            "word_count": metadata.get("word_count", 0),
            "entities": metadata.get("entities", "").split(",") if metadata.get("entities") else [],
            "related_topics": metadata.get("related_topics", "").split(",") if metadata.get("related_topics") else []
        }

        # Enrich with graph data if available
        if self.graph:
            node = self.graph.get_node(article_id)
            if node:
                # Get related entities from graph
                entities = self.graph.neighbors(article_id, direction="out", edge_type="MENTIONS")
                if entities:
                    article["mentioned_entities"] = [
                        {"name": e.properties.get("name"), "type": e.properties.get("type")}
                        for e in entities
                    ]

                # Get source info
                sources = self.graph.neighbors(article_id, direction="out", edge_type="PUBLISHED_BY")
                if sources:
                    article["publisher"] = sources[0].properties.get("name")

                # Get topics from graph
                topics = self.graph.neighbors(article_id, direction="out", edge_type="ABOUT")
                if topics:
                    article["topics_graph"] = [t.properties.get("name") for t in topics]

        return article

    def print_article(self, article: dict):
        """Pretty print a full article."""
        if "error" in article:
            print(f"\n  Error: {article['error']}")
            return

        print(f"\n{'='*70}")
        print(f" {article.get('headline', 'N/A')}")
        print(f"{'='*70}")
        print(f"\n  ID:        {article.get('id')}")
        print(f"  Source:    {article.get('publisher') or article.get('source')}")
        print(f"  Date:      {article.get('published_date')}")
        print(f"  Category:  {article.get('category')}")
        print(f"  Topic:     {article.get('topic')}")
        print(f"  Sentiment: {article.get('sentiment')}")
        print(f"  Words:     {article.get('word_count')}")

        if article.get('mentioned_entities'):
            print(f"\n  Entities Mentioned:")
            for e in article['mentioned_entities'][:5]:
                print(f"    - {e['name']} ({e['type']})")

        if article.get('related_topics'):
            print(f"\n  Related Topics: {', '.join(article['related_topics'][:5])}")

        print(f"\n  Content:")
        print(f"  {'-'*66}")
        # Word wrap the content
        content = article.get('content', 'N/A')
        words = content.split()
        line = "  "
        for word in words:
            if len(line) + len(word) > 68:
                print(line)
                line = "  " + word + " "
            else:
                line += word + " "
        if line.strip():
            print(line)
        print()

    # =========================================================================
    # GRAPH-BASED QUERIES
    # =========================================================================

    def find_entity(self, entity_name: str) -> dict:
        """
        Find information about an entity (company, person, etc.)

        Returns articles mentioning the entity, related entities, etc.
        """
        if not self.graph:
            return {"error": "Graph not available"}

        # Normalize entity name to ID format
        entity_id = f"entity_{entity_name.replace(' ', '_').lower()}"
        node = self.graph.get_node(entity_id)

        if not node:
            # Try partial match
            entities = self.graph.find_nodes(label="Entity")
            matches = [e for e in entities
                      if entity_name.lower() in e.properties.get('name', '').lower()]
            if matches:
                node = matches[0]
                entity_id = node.id
            else:
                return {"error": f"Entity '{entity_name}' not found"}

        result = {
            "name": node.properties.get('name'),
            "type": node.properties.get('type'),
            "articles": [],
            "related_entities": [],
            "sentiment_summary": {}
        }

        # Find articles mentioning this entity
        articles = self.graph.neighbors(entity_id, direction="in", edge_type="MENTIONS")
        result["article_count"] = len(articles)

        # Get sample articles
        for article in articles[:5]:
            result["articles"].append({
                "headline": article.properties.get('headline', '')[:100],
                "category": article.properties.get('category'),
                "date": article.properties.get('date'),
                "sentiment": article.properties.get('sentiment')
            })

        # Analyze sentiment
        sentiments = [a.properties.get('sentiment', 'neutral') for a in articles]
        result["sentiment_summary"] = {s: sentiments.count(s) for s in set(sentiments)}

        # Find related entities via hyperedges (co-mentions)
        hyperedges = self.graph.get_hyperedges_by_node(entity_id)
        co_mentioned = set()
        for he in hyperedges[:50]:
            for nid in he.nodes:
                if nid != entity_id:
                    n = self.graph.get_node(nid)
                    if n:
                        co_mentioned.add(n.properties.get('name', nid))
        result["related_entities"] = list(co_mentioned)[:10]

        # If company, find leader
        if node.properties.get('type') == 'Company':
            leaders = self.graph.neighbors(entity_id, direction="in", edge_type="LEADS")
            if leaders:
                result["leaders"] = [l.properties.get('name') for l in leaders]

            locations = self.graph.neighbors(entity_id, direction="out", edge_type="HEADQUARTERED_IN")
            if locations:
                result["headquarters"] = [l.properties.get('name') for l in locations]

        return result

    def find_topic_articles(self, topic: str, k: int = 10) -> list:
        """Find articles about a specific topic using the graph."""
        if not self.graph:
            # Fall back to vector search
            return self.search(topic, k=k)

        # Find topic node
        topic_id = f"topic_{topic.replace(' ', '_').lower()}"
        node = self.graph.get_node(topic_id)

        if not node:
            # Try partial match or fall back to search
            return self.search(topic, k=k)

        # Get articles linked to this topic
        articles = self.graph.neighbors(topic_id, direction="in", edge_type="ABOUT")

        results = []
        for article in articles[:k]:
            results.append({
                "id": article.id,
                "headline": article.properties.get('headline'),
                "category": article.properties.get('category'),
                "source": article.properties.get('source'),
                "date": article.properties.get('date'),
                "sentiment": article.properties.get('sentiment')
            })

        return results

    def find_source_articles(self, source_name: str, k: int = 10) -> list:
        """Find articles from a specific news source."""
        if not self.graph:
            return self.search("news", k=k, source=source_name)

        source_id = f"source_{source_name.replace(' ', '_').lower()}"
        node = self.graph.get_node(source_id)

        if not node:
            sources = self.graph.find_nodes(label="Source")
            matches = [s for s in sources
                      if source_name.lower() in s.properties.get('name', '').lower()]
            if matches:
                node = matches[0]
                source_id = node.id
            else:
                return []

        articles = self.graph.neighbors(source_id, direction="in", edge_type="PUBLISHED_BY")

        results = []
        for article in articles[:k]:
            results.append({
                "id": article.id,
                "headline": article.properties.get('headline'),
                "category": article.properties.get('category'),
                "date": article.properties.get('date'),
                "sentiment": article.properties.get('sentiment')
            })

        return results

    def get_category_stats(self) -> dict:
        """Get statistics about article categories."""
        if not self.graph:
            return {"error": "Graph not available"}

        categories = self.graph.find_nodes(label="Category")
        stats = {}

        for cat in categories:
            name = cat.properties.get('name')
            articles = self.graph.neighbors(cat.id, direction="in", edge_type="IN_CATEGORY")
            stats[name] = len(articles)

        return stats

    def get_trending_entities(self, k: int = 10) -> list:
        """Get the most mentioned entities."""
        if not self.graph:
            return []

        entities = self.graph.find_nodes(label="Entity")

        entity_counts = []
        for entity in entities:
            mentions = self.graph.neighbors(entity.id, direction="in", edge_type="MENTIONS")
            entity_counts.append({
                "name": entity.properties.get('name'),
                "type": entity.properties.get('type'),
                "mention_count": len(mentions)
            })

        # Sort by mentions
        entity_counts.sort(key=lambda x: x['mention_count'], reverse=True)
        return entity_counts[:k]

    # =========================================================================
    # DISPLAY HELPERS
    # =========================================================================

    def print_results(self, results: list, title: str = "Search Results"):
        """Pretty print search results."""
        print(f"\n{'='*60}")
        print(f" {title}")
        print(f"{'='*60}")

        if not results:
            print("  No results found.")
            return

        # Store last results for quick access
        self._last_results = results

        for i, r in enumerate(results, 1):
            if hasattr(r, 'metadata'):
                # SearchResult object
                article_id = r.id
                headline = r.metadata.get('headline', 'N/A')[:60]
                category = r.metadata.get('category', 'N/A')
                source = r.metadata.get('source', 'N/A')
                sentiment = r.metadata.get('sentiment', 'N/A')
                score = f"{r.score:.4f}"
                print(f"\n  {i}. [{score}] {headline}...")
                print(f"     ID: {article_id}")
                print(f"     Category: {category} | Source: {source} | Sentiment: {sentiment}")
            elif isinstance(r, dict):
                # Dictionary result
                article_id = r.get('id', 'N/A')
                headline = r.get('headline', 'N/A')[:60] if r.get('headline') else 'N/A'
                print(f"\n  {i}. {headline}...")
                print(f"     ID: {article_id}")
                if 'category' in r:
                    print(f"     Category: {r.get('category')} | Source: {r.get('source', 'N/A')}")

        print(f"\n  TIP: Use 'read <ID>' or 'read {len(results)}' to read an article")
        print()

    def get_result_by_number(self, number: int):
        """Get article ID from last search results by number."""
        if not hasattr(self, '_last_results') or not self._last_results:
            return None
        if 1 <= number <= len(self._last_results):
            result = self._last_results[number - 1]
            if hasattr(result, 'id'):
                return result.id
            elif isinstance(result, dict):
                return result.get('id')
        return None

    def print_entity_info(self, info: dict):
        """Pretty print entity information."""
        if 'error' in info:
            print(f"\n  Error: {info['error']}")
            return

        print(f"\n{'='*60}")
        print(f" Entity: {info['name']} ({info['type']})")
        print(f"{'='*60}")

        print(f"\n  Total Articles Mentioning: {info['article_count']}")

        if info.get('leaders'):
            print(f"  Leaders: {', '.join(info['leaders'])}")
        if info.get('headquarters'):
            print(f"  Headquarters: {', '.join(info['headquarters'])}")

        print(f"\n  Sentiment Distribution:")
        for sent, count in info['sentiment_summary'].items():
            print(f"    - {sent}: {count}")

        if info['related_entities']:
            print(f"\n  Frequently Co-mentioned With:")
            print(f"    {', '.join(info['related_entities'][:5])}")

        if info['articles']:
            print(f"\n  Sample Articles:")
            for article in info['articles'][:3]:
                print(f"    - {article['headline'][:60]}...")
                print(f"      ({article['category']}, {article['sentiment']})")
        print()


def interactive_mode(query_interface: NewsQueryInterface):
    """Run an interactive query session."""
    print("\n" + "="*60)
    print(" NEWS INTELLIGENCE - INTERACTIVE QUERY MODE")
    print("="*60)
    print("""
Commands:
  search <query>         - Semantic search for articles
  read <id or number>    - Read a full article (by ID or result #)
  entity <name>          - Look up an entity (company, person)
  topic <name>           - Find articles about a topic
  source <name>          - Find articles from a source
  category <name>        - Browse articles by category
  trending               - Show trending entities
  stats                  - Show category statistics
  help                   - Show this help
  quit                   - Exit

Examples:
  search artificial intelligence
  read 3                              <- read 3rd result from search
  read article_00000001               <- read by ID
  entity Apple
  topic climate change
  source TechCrunch
    """)

    while True:
        try:
            user_input = input("\n> ").strip()

            if not user_input:
                continue

            parts = user_input.split(maxsplit=1)
            command = parts[0].lower()
            args = parts[1] if len(parts) > 1 else ""

            if command == "quit" or command == "exit":
                print("Goodbye!")
                break

            elif command == "help":
                print("Commands: search, read, entity, topic, source, category, trending, stats, quit")

            elif command == "read":
                if not args:
                    print("Usage: read <article_id> or read <number>")
                    print("Examples:")
                    print("  read article_00000001")
                    print("  read 3  (reads 3rd result from last search)")
                    continue

                # Check if it's a number (read from last results)
                try:
                    result_num = int(args)
                    article_id = query_interface.get_result_by_number(result_num)
                    if not article_id:
                        print(f"  No result #{result_num} in last search. Run a search first.")
                        continue
                except ValueError:
                    # It's an article ID
                    article_id = args

                article = query_interface.read_article(article_id)
                query_interface.print_article(article)

            elif command == "search":
                if not args:
                    print("Usage: search <query>")
                    continue
                results = query_interface.search(args, k=10)
                query_interface.print_results(results, f"Search: '{args}'")

            elif command == "entity":
                if not args:
                    print("Usage: entity <name>")
                    continue
                info = query_interface.find_entity(args)
                query_interface.print_entity_info(info)

            elif command == "topic":
                if not args:
                    print("Usage: topic <name>")
                    continue
                results = query_interface.find_topic_articles(args, k=10)
                query_interface.print_results(results, f"Topic: '{args}'")

            elif command == "source":
                if not args:
                    print("Usage: source <name>")
                    continue
                results = query_interface.find_source_articles(args, k=10)
                query_interface.print_results(results, f"Source: '{args}'")

            elif command == "category":
                if not args:
                    print("Usage: category <name>")
                    continue
                results = query_interface.search_by_category(args, k=10)
                query_interface.print_results(results, f"Category: '{args}'")

            elif command == "trending":
                entities = query_interface.get_trending_entities(k=10)
                print("\n" + "="*60)
                print(" TRENDING ENTITIES")
                print("="*60)
                for i, e in enumerate(entities, 1):
                    print(f"  {i}. {e['name']} ({e['type']}): {e['mention_count']} mentions")

            elif command == "stats":
                stats = query_interface.get_category_stats()
                print("\n" + "="*60)
                print(" CATEGORY STATISTICS")
                print("="*60)
                if isinstance(stats, dict) and 'error' not in stats:
                    for cat, count in sorted(stats.items(), key=lambda x: x[1], reverse=True):
                        print(f"  {cat}: {count:,} articles")
                else:
                    print(f"  {stats}")

            else:
                # Treat as search query
                results = query_interface.search(user_input, k=10)
                query_interface.print_results(results, f"Search: '{user_input}'")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Query the News Intelligence Database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python query_news_db.py                         # Interactive mode
  python query_news_db.py --search "AI trends"    # Quick search
  python query_news_db.py --entity "Google"       # Entity lookup
  python query_news_db.py --topic "climate"       # Topic articles
  python query_news_db.py --trending              # Trending entities
        """
    )

    parser.add_argument("--db-path", default="./news_intelligence_db",
                       help="Path to the database")
    parser.add_argument("--search", "-s", type=str,
                       help="Search for articles")
    parser.add_argument("--read", "-r", type=str,
                       help="Read a specific article by ID")
    parser.add_argument("--entity", "-e", type=str,
                       help="Look up an entity")
    parser.add_argument("--topic", "-t", type=str,
                       help="Find articles about a topic")
    parser.add_argument("--source", type=str,
                       help="Find articles from a source")
    parser.add_argument("--category", "-c", type=str,
                       help="Filter by category")
    parser.add_argument("--sentiment", type=str,
                       choices=["positive", "negative", "neutral"],
                       help="Filter by sentiment")
    parser.add_argument("--trending", action="store_true",
                       help="Show trending entities")
    parser.add_argument("--stats", action="store_true",
                       help="Show category statistics")
    parser.add_argument("-k", type=int, default=10,
                       help="Number of results (default: 10)")

    args = parser.parse_args()

    # Load database
    try:
        qi = NewsQueryInterface(args.db_path)
    except Exception as e:
        print(f"Error loading database: {e}")
        print("\nMake sure you've run news_intelligence_demo.py first!")
        return

    # Handle command-line queries
    if args.search:
        results = qi.search(args.search, k=args.k,
                           category=args.category,
                           sentiment=args.sentiment)
        qi.print_results(results, f"Search: '{args.search}'")

    elif args.read:
        article = qi.read_article(args.read)
        qi.print_article(article)

    elif args.entity:
        info = qi.find_entity(args.entity)
        qi.print_entity_info(info)

    elif args.topic:
        results = qi.find_topic_articles(args.topic, k=args.k)
        qi.print_results(results, f"Topic: '{args.topic}'")

    elif args.source:
        results = qi.find_source_articles(args.source, k=args.k)
        qi.print_results(results, f"Source: '{args.source}'")

    elif args.trending:
        entities = qi.get_trending_entities(k=args.k)
        print("\n" + "="*60)
        print(" TRENDING ENTITIES")
        print("="*60)
        for i, e in enumerate(entities, 1):
            print(f"  {i}. {e['name']} ({e['type']}): {e['mention_count']} mentions")

    elif args.stats:
        stats = qi.get_category_stats()
        print("\n" + "="*60)
        print(" CATEGORY STATISTICS")
        print("="*60)
        for cat, count in sorted(stats.items(), key=lambda x: x[1], reverse=True):
            print(f"  {cat}: {count:,} articles")

    else:
        # Interactive mode
        interactive_mode(qi)


if __name__ == "__main__":
    main()
