#!/usr/bin/env python3
"""
================================================================================
NEWS INTELLIGENCE & RESEARCH PLATFORM
================================================================================

A practical real-world use case demonstrating PyVectorDB capabilities with
massive data (100,000+ articles). This script simulates a news intelligence
platform that allows:

1. Semantic search across articles
2. Filtered queries by category, date, source, sentiment
3. Hybrid keyword + vector search
4. Topic clustering and related article discovery
5. Knowledge graph for entity relationships
6. Performance benchmarking at scale

Use Cases:
- News aggregation platforms (Google News, Apple News)
- Research intelligence tools (Factiva, LexisNexis)
- Media monitoring services (Meltwater, Cision)
- Corporate intelligence & due diligence
- Academic research assistants

Author: Generated for custom-python-vectordb demonstration
Date: December 2025
================================================================================
"""

import numpy as np
import time
import random
import hashlib
import json
import os
import sys
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
import threading
import concurrent.futures

# Add the project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import PyVectorDB components
from vectordb_optimized import VectorDB, Filter, DistanceMetric

# Optional imports with fallbacks
try:
    from quantization import ScalarQuantizer, BinaryQuantizer, ProductQuantizer
    HAS_QUANTIZATION = True
except ImportError:
    HAS_QUANTIZATION = False
    print("Warning: Quantization module not available")

try:
    from parallel_search import ParallelSearchEngine, MemoryMappedVectors
    HAS_PARALLEL = True
except ImportError:
    HAS_PARALLEL = False
    print("Warning: Parallel search module not available")

try:
    from graph import GraphDB, NodeBuilder, EdgeBuilder, HyperedgeBuilder
    from hybrid_graph_vector import HybridGraphVectorDB, GraphVectorSearchResult
    HAS_GRAPH = True
except ImportError:
    HAS_GRAPH = False
    print("Warning: Graph module not available")

try:
    from hybrid_search import HybridCollection
    HAS_HYBRID_SEARCH = True
except ImportError:
    HAS_HYBRID_SEARCH = False
    print("Warning: Hybrid search module not available")


# ================================================================================
# CONFIGURATION
# ================================================================================

@dataclass
class Config:
    """Configuration for the news intelligence demo."""

    # Dataset size - adjust for your hardware
    num_articles: int = 100_000  # Total articles to generate
    embedding_dim: int = 384      # Embedding dimensions (matches all-MiniLM-L6-v2)

    # Database settings
    db_path: str = "./news_intelligence_db"
    collection_name: str = "articles"

    # HNSW tuning for large datasets
    hnsw_m: int = 32              # Higher M = better recall, more memory
    hnsw_ef_construction: int = 200
    hnsw_ef_search: int = 100

    # Batch processing
    batch_size: int = 5000        # Articles per batch insert

    # Parallel processing
    num_workers: int = 8          # CPU cores for parallel search

    # Time range for articles
    start_date: datetime = field(default_factory=lambda: datetime(2024, 1, 1))
    end_date: datetime = field(default_factory=lambda: datetime(2025, 12, 5))

    # Search demo settings
    demo_queries: int = 100       # Number of demo searches


# ================================================================================
# NEWS DATA GENERATION
# ================================================================================

# Real-world news categories and topics based on 2024-2025 events
NEWS_CATEGORIES = {
    "technology": {
        "weight": 0.20,
        "topics": [
            "artificial intelligence", "machine learning", "OpenAI GPT-5", "Claude AI",
            "autonomous vehicles", "Waymo expansion", "Tesla FSD", "robotics",
            "quantum computing", "cybersecurity", "ransomware attacks",
            "semiconductor shortage", "Nvidia chips", "AMD processors", "Intel foundry",
            "cloud computing", "AWS", "Azure", "Google Cloud",
            "blockchain", "cryptocurrency", "Bitcoin ETF", "Ethereum staking",
            "5G networks", "6G research", "satellite internet", "Starlink",
            "virtual reality", "augmented reality", "Apple Vision Pro", "Meta Quest",
            "edge computing", "IoT devices", "smart cities", "digital twins"
        ],
        "sources": ["TechCrunch", "Wired", "The Verge", "Ars Technica", "MIT Technology Review",
                   "CNET", "Engadget", "ZDNet", "VentureBeat", "IEEE Spectrum"]
    },
    "business": {
        "weight": 0.18,
        "topics": [
            "stock market", "S&P 500", "NASDAQ", "Dow Jones",
            "Federal Reserve", "interest rates", "inflation", "recession fears",
            "corporate earnings", "quarterly reports", "IPO market",
            "mergers acquisitions", "antitrust regulation", "Big Tech breakup",
            "startup funding", "venture capital", "unicorn valuations",
            "supply chain", "logistics disruption", "manufacturing reshoring",
            "real estate market", "commercial property", "housing crisis",
            "employment report", "job market", "remote work trends",
            "ESG investing", "sustainable finance", "green bonds"
        ],
        "sources": ["Wall Street Journal", "Financial Times", "Bloomberg", "Reuters",
                   "CNBC", "Forbes", "Fortune", "Business Insider", "The Economist"]
    },
    "politics": {
        "weight": 0.15,
        "topics": [
            "US elections", "presidential campaign", "congressional vote",
            "Trump administration", "executive orders", "tariff policy",
            "China relations", "trade war", "sanctions",
            "Russia Ukraine", "NATO alliance", "defense spending",
            "Middle East peace", "Gaza ceasefire", "Israel Hamas",
            "immigration policy", "border security", "refugee crisis",
            "climate legislation", "Paris Agreement", "carbon tax",
            "healthcare reform", "Medicare expansion", "drug pricing",
            "Supreme Court", "judicial appointments", "constitutional law"
        ],
        "sources": ["Washington Post", "New York Times", "Politico", "The Hill",
                   "Associated Press", "NPR", "BBC News", "Al Jazeera"]
    },
    "science": {
        "weight": 0.12,
        "topics": [
            "space exploration", "NASA Artemis", "SpaceX Mars", "James Webb telescope",
            "climate change", "global warming", "extreme weather", "sea level rise",
            "renewable energy", "solar power", "wind farms", "nuclear fusion",
            "medical research", "cancer treatment", "gene therapy", "CRISPR",
            "pandemic preparedness", "vaccine development", "public health",
            "biodiversity loss", "species extinction", "conservation",
            "ocean science", "marine biology", "coral reefs",
            "particle physics", "CERN", "dark matter", "gravitational waves"
        ],
        "sources": ["Nature", "Science", "Scientific American", "New Scientist",
                   "National Geographic", "Space.com", "LiveScience", "Phys.org"]
    },
    "world": {
        "weight": 0.15,
        "topics": [
            "European Union", "Brexit aftermath", "EU elections",
            "China economy", "Belt and Road", "Taiwan strait",
            "India growth", "Modi government", "tech sector",
            "Africa development", "African Union", "economic partnerships",
            "Latin America", "Brazil politics", "Argentina economy",
            "Southeast Asia", "ASEAN summit", "regional trade",
            "United Nations", "Security Council", "peacekeeping",
            "humanitarian crisis", "disaster relief", "international aid"
        ],
        "sources": ["BBC World", "Al Jazeera", "DW News", "France 24",
                   "The Guardian", "South China Morning Post", "Japan Times"]
    },
    "health": {
        "weight": 0.10,
        "topics": [
            "mental health", "depression treatment", "anxiety disorders",
            "obesity epidemic", "weight loss drugs", "Ozempic",
            "heart disease", "cardiovascular health", "cholesterol",
            "diabetes management", "insulin costs", "glucose monitoring",
            "aging research", "longevity science", "anti-aging",
            "nutrition science", "diet trends", "supplements",
            "fitness technology", "wearable devices", "health tracking",
            "healthcare costs", "insurance coverage", "hospital systems"
        ],
        "sources": ["WebMD", "Healthline", "Medical News Today", "STAT News",
                   "The Lancet", "JAMA Network", "New England Journal"]
    },
    "sports": {
        "weight": 0.05,
        "topics": [
            "NFL football", "Super Bowl", "quarterback trades",
            "NBA basketball", "playoff race", "All-Star game",
            "soccer World Cup", "Champions League", "transfer market",
            "Olympics 2024", "Paris games", "medal count",
            "tennis Grand Slam", "golf majors", "Formula 1",
            "esports", "gaming tournaments", "streaming platforms"
        ],
        "sources": ["ESPN", "Sports Illustrated", "Bleacher Report", "The Athletic",
                   "Sky Sports", "NBC Sports", "Fox Sports"]
    },
    "entertainment": {
        "weight": 0.05,
        "topics": [
            "streaming wars", "Netflix", "Disney Plus", "HBO Max",
            "movie releases", "box office", "film festivals",
            "music industry", "concert tours", "album releases",
            "celebrity news", "awards shows", "Oscars Grammy",
            "video games", "gaming industry", "console releases",
            "social media", "TikTok trends", "influencer culture"
        ],
        "sources": ["Variety", "Hollywood Reporter", "Entertainment Weekly",
                   "Billboard", "Rolling Stone", "IGN", "Polygon"]
    }
}

# Sentiment categories
SENTIMENTS = ["positive", "negative", "neutral", "mixed"]
SENTIMENT_WEIGHTS = [0.25, 0.25, 0.35, 0.15]

# Article templates for realistic text generation
HEADLINE_TEMPLATES = [
    "{topic} Shows Promising Growth in Latest Report",
    "Breaking: Major Development in {topic} Sector",
    "{source} Reports Significant Changes in {topic}",
    "Analysis: What {topic} Means for the Future",
    "Experts Weigh In on {topic} Trends",
    "{topic} Faces New Challenges Amid Global Uncertainty",
    "Inside the Revolution: How {topic} is Changing Everything",
    "The Rise and Fall of {topic}: A Comprehensive Look",
    "{topic} Update: Key Developments You Need to Know",
    "Opinion: Why {topic} Matters More Than Ever",
    "Investigation: The Hidden Story Behind {topic}",
    "{topic} Market Reaches New Milestone",
    "Industry Leaders Discuss Future of {topic}",
    "How {topic} is Reshaping the Global Landscape",
    "{topic} in Crisis: What Went Wrong?"
]

CONTENT_TEMPLATES = [
    """In a significant development for {topic}, industry analysts are reporting substantial changes
    that could reshape the landscape. According to sources close to the matter, recent trends indicate
    a shift in how stakeholders approach key challenges. Experts from {source} suggest that these
    developments warrant close attention from investors and policymakers alike. The implications
    extend beyond immediate concerns, potentially affecting related sectors and long-term strategic
    planning. Market observers note that similar patterns have emerged in comparable situations,
    though the current circumstances present unique characteristics that distinguish them from
    historical precedents.""",

    """The latest analysis of {topic} reveals complex dynamics at play, with multiple factors
    contributing to the current state of affairs. Research conducted by leading institutions
    highlights both opportunities and risks that stakeholders must carefully evaluate. Industry
    veterans point to historical context as essential for understanding present developments,
    while younger analysts emphasize the unprecedented nature of certain aspects. The debate
    continues among experts about optimal strategies for navigating this evolving situation.
    Meanwhile, affected parties are adapting their approaches in response to changing conditions.""",

    """Recent events surrounding {topic} have captured widespread attention from observers across
    various sectors. The convergence of technological, economic, and social factors has created
    a unique environment for innovation and disruption. Stakeholders are reassessing their
    positions in light of new information and shifting market dynamics. Commentary from {source}
    indicates growing interest among both institutional and retail participants. The trajectory
    of these developments remains subject to multiple variables, including regulatory decisions,
    competitive pressures, and broader macroeconomic trends that continue to evolve.""",

    """A comprehensive review of {topic} demonstrates the multifaceted nature of contemporary
    challenges and opportunities in this space. Data from multiple sources confirms patterns
    that have been emerging over recent periods, while also revealing unexpected variations
    that merit further investigation. Industry leaders are calling for collaborative approaches
    to address systemic issues, even as competitive dynamics intensify. The intersection of
    traditional practices with emerging technologies creates both friction and potential for
    breakthrough innovations. Observers anticipate continued volatility as the sector matures.""",

    """The ongoing evolution of {topic} presents a compelling case study in adaptation and
    resilience. Market participants have demonstrated remarkable flexibility in responding to
    disruptions, though questions remain about long-term sustainability. Analysis from {source}
    suggests that current trends may accelerate, potentially reaching inflection points that
    could fundamentally alter existing paradigms. Regulatory frameworks are being tested and
    refined in response to novel challenges, while international coordination efforts face
    familiar obstacles. The outlook depends heavily on decisions made in the coming months."""
]

# Named entities for knowledge graph
ENTITIES = {
    "companies": [
        "Apple", "Google", "Microsoft", "Amazon", "Meta", "Tesla", "Nvidia",
        "OpenAI", "Anthropic", "Samsung", "Intel", "AMD", "IBM", "Oracle",
        "Salesforce", "Adobe", "Netflix", "Disney", "JPMorgan", "Goldman Sachs"
    ],
    "people": [
        "Elon Musk", "Tim Cook", "Satya Nadella", "Sam Altman", "Sundar Pichai",
        "Mark Zuckerberg", "Jensen Huang", "Jamie Dimon", "Warren Buffett",
        "Dario Amodei", "Jeff Bezos", "Andy Jassy", "Lisa Su", "Pat Gelsinger"
    ],
    "locations": [
        "Silicon Valley", "New York", "Washington DC", "Beijing", "London",
        "Tokyo", "Brussels", "Singapore", "Mumbai", "Tel Aviv", "Berlin",
        "San Francisco", "Seattle", "Austin", "Boston", "Shanghai"
    ],
    "organizations": [
        "Federal Reserve", "SEC", "FTC", "European Commission", "NATO",
        "United Nations", "World Bank", "IMF", "WHO", "WTO", "OPEC"
    ]
}


@dataclass
class Article:
    """Represents a news article with metadata."""
    id: str
    headline: str
    content: str
    category: str
    topic: str
    source: str
    published_date: datetime
    sentiment: str
    word_count: int
    read_time_minutes: int
    entities: List[str]
    related_topics: List[str]
    embedding: Optional[np.ndarray] = None

    def to_metadata(self) -> Dict:
        """Convert to metadata dict for storage."""
        return {
            "headline": self.headline,
            "content": self.content[:500] + "..." if len(self.content) > 500 else self.content,
            "category": self.category,
            "topic": self.topic,
            "source": self.source,
            "published_date": self.published_date.isoformat(),
            "published_year": self.published_date.year,
            "published_month": self.published_date.month,
            "sentiment": self.sentiment,
            "word_count": self.word_count,
            "read_time_minutes": self.read_time_minutes,
            "entities": ",".join(self.entities),
            "related_topics": ",".join(self.related_topics)
        }


class NewsDataGenerator:
    """Generates realistic synthetic news articles with embeddings."""

    def __init__(self, config: Config, seed: int = 42):
        self.config = config
        self.rng = np.random.default_rng(seed)
        random.seed(seed)

        # Pre-compute category weights
        self.categories = list(NEWS_CATEGORIES.keys())
        self.category_weights = [NEWS_CATEGORIES[c]["weight"] for c in self.categories]

        # Topic embeddings cache (simulate learned topic representations)
        self.topic_embeddings = {}
        self._initialize_topic_embeddings()

    def _initialize_topic_embeddings(self):
        """Create base embeddings for each topic (simulates training)."""
        print("Initializing topic embedding space...")
        for category, data in NEWS_CATEGORIES.items():
            # Create a category centroid
            category_base = self.rng.standard_normal(self.config.embedding_dim).astype(np.float32)
            category_base = category_base / np.linalg.norm(category_base)

            for topic in data["topics"]:
                # Each topic is a perturbation of the category centroid
                noise = self.rng.standard_normal(self.config.embedding_dim).astype(np.float32) * 0.3
                topic_emb = category_base + noise
                topic_emb = topic_emb / np.linalg.norm(topic_emb)
                self.topic_embeddings[topic] = topic_emb

    def _generate_embedding(self, topic: str, content: str) -> np.ndarray:
        """Generate a realistic embedding for an article."""
        # Start with topic embedding
        base_emb = self.topic_embeddings.get(topic,
                   self.rng.standard_normal(self.config.embedding_dim).astype(np.float32))

        # Add content-based variation (simulates actual content differences)
        content_hash = int(hashlib.md5(content.encode()).hexdigest(), 16)
        content_rng = np.random.default_rng(content_hash % (2**32))
        content_noise = content_rng.standard_normal(self.config.embedding_dim).astype(np.float32) * 0.15

        # Combine and normalize
        embedding = base_emb + content_noise
        embedding = embedding / np.linalg.norm(embedding)

        return embedding

    def _generate_random_date(self) -> datetime:
        """Generate a random date within the configured range."""
        delta = self.config.end_date - self.config.start_date
        random_days = random.randint(0, delta.days)
        random_seconds = random.randint(0, 86400)
        return self.config.start_date + timedelta(days=random_days, seconds=random_seconds)

    def _select_entities(self, topic: str) -> List[str]:
        """Select relevant entities for an article."""
        entities = []

        # Select 1-4 entities based on topic relevance
        num_entities = random.randint(1, 4)

        # Weight entity types by category
        if "tech" in topic.lower() or "ai" in topic.lower():
            entity_pool = ENTITIES["companies"] + ENTITIES["people"]
        elif "politic" in topic.lower() or "government" in topic.lower():
            entity_pool = ENTITIES["organizations"] + ENTITIES["locations"] + ENTITIES["people"]
        else:
            entity_pool = (ENTITIES["companies"] + ENTITIES["people"] +
                          ENTITIES["locations"] + ENTITIES["organizations"])

        entities = random.sample(entity_pool, min(num_entities, len(entity_pool)))
        return entities

    def _select_related_topics(self, primary_topic: str, category: str) -> List[str]:
        """Select related topics for cross-referencing."""
        all_topics = NEWS_CATEGORIES[category]["topics"]
        related = [t for t in all_topics if t != primary_topic]
        return random.sample(related, min(3, len(related)))

    def generate_article(self, article_id: int) -> Article:
        """Generate a single news article."""
        # Select category and topic
        category = random.choices(self.categories, weights=self.category_weights)[0]
        category_data = NEWS_CATEGORIES[category]
        topic = random.choice(category_data["topics"])
        source = random.choice(category_data["sources"])

        # Generate headline and content
        headline_template = random.choice(HEADLINE_TEMPLATES)
        headline = headline_template.format(topic=topic.title(), source=source)

        content_template = random.choice(CONTENT_TEMPLATES)
        content = content_template.format(topic=topic, source=source)

        # Calculate metrics
        word_count = len(content.split())
        read_time = max(1, word_count // 200)

        # Generate other attributes
        published_date = self._generate_random_date()
        sentiment = random.choices(SENTIMENTS, weights=SENTIMENT_WEIGHTS)[0]
        entities = self._select_entities(topic)
        related_topics = self._select_related_topics(topic, category)

        # Generate unique ID
        article_id_str = f"article_{article_id:08d}"

        # Generate embedding
        embedding = self._generate_embedding(topic, content)

        return Article(
            id=article_id_str,
            headline=headline,
            content=content,
            category=category,
            topic=topic,
            source=source,
            published_date=published_date,
            sentiment=sentiment,
            word_count=word_count,
            read_time_minutes=read_time,
            entities=entities,
            related_topics=related_topics,
            embedding=embedding
        )

    def generate_batch(self, start_id: int, count: int) -> List[Article]:
        """Generate a batch of articles."""
        return [self.generate_article(start_id + i) for i in range(count)]

    def generate_all(self, show_progress: bool = True) -> List[Article]:
        """Generate all articles."""
        articles = []
        total = self.config.num_articles
        batch_size = self.config.batch_size

        print(f"\nGenerating {total:,} news articles...")
        start_time = time.time()

        for batch_start in range(0, total, batch_size):
            batch_count = min(batch_size, total - batch_start)
            batch = self.generate_batch(batch_start, batch_count)
            articles.extend(batch)

            if show_progress:
                progress = (batch_start + batch_count) / total * 100
                elapsed = time.time() - start_time
                rate = (batch_start + batch_count) / elapsed
                remaining = (total - batch_start - batch_count) / rate if rate > 0 else 0
                print(f"\r  Progress: {progress:.1f}% ({batch_start + batch_count:,}/{total:,}) "
                      f"| Rate: {rate:.0f} articles/sec | ETA: {remaining:.0f}s", end="")

        print(f"\n  Generated {len(articles):,} articles in {time.time() - start_time:.1f}s")
        return articles


# ================================================================================
# DATABASE OPERATIONS
# ================================================================================

class NewsIntelligenceDB:
    """News intelligence database with vector search capabilities."""

    def __init__(self, config: Config):
        self.config = config
        self.db = None
        self.collection = None
        self.graph = None
        self.parallel_engine = None
        self.quantizer = None

    def initialize(self):
        """Initialize the database and collection."""
        print(f"\nInitializing database at {self.config.db_path}...")

        # Create database
        self.db = VectorDB(self.config.db_path)

        # Create or get collection
        try:
            self.collection = self.db.get_collection(self.config.collection_name)
            print(f"  Loaded existing collection: {self.config.collection_name}")
            print(f"  Existing articles: {self.collection.count():,}")
        except:
            self.collection = self.db.create_collection(
                name=self.config.collection_name,
                dimensions=self.config.embedding_dim,
                metric="cosine",
                M=self.config.hnsw_m,
                ef_construction=self.config.hnsw_ef_construction,
                ef_search=self.config.hnsw_ef_search
            )
            print(f"  Created new collection: {self.config.collection_name}")

        # Initialize optional components
        if HAS_PARALLEL:
            self.parallel_engine = ParallelSearchEngine(n_workers=self.config.num_workers)
            print(f"  Parallel search engine initialized with {self.config.num_workers} workers")

        if HAS_QUANTIZATION:
            self.quantizer = ScalarQuantizer(dimensions=self.config.embedding_dim)
            print("  Scalar quantizer initialized")

        if HAS_GRAPH:
            graph_path = os.path.join(self.config.db_path, "graph")
            self.graph = GraphDB(graph_path)
            print(f"  Knowledge graph initialized at {graph_path}")

    def ingest_articles(self, articles: List[Article], show_progress: bool = True):
        """Ingest articles into the database."""
        print(f"\nIngesting {len(articles):,} articles into database...")
        start_time = time.time()

        batch_size = self.config.batch_size
        total = len(articles)

        for batch_start in range(0, total, batch_size):
            batch_end = min(batch_start + batch_size, total)
            batch = articles[batch_start:batch_end]

            # Prepare batch data
            vectors = np.array([a.embedding for a in batch], dtype=np.float32)
            ids = [a.id for a in batch]
            metadata_list = [a.to_metadata() for a in batch]

            # Batch insert
            self.collection.insert_batch(vectors, ids, metadata_list)

            if show_progress:
                progress = batch_end / total * 100
                elapsed = time.time() - start_time
                rate = batch_end / elapsed
                remaining = (total - batch_end) / rate if rate > 0 else 0
                print(f"\r  Progress: {progress:.1f}% ({batch_end:,}/{total:,}) "
                      f"| Rate: {rate:.0f} articles/sec | ETA: {remaining:.0f}s", end="")

        # Save database
        self.db.save()

        elapsed = time.time() - start_time
        print(f"\n  Ingested {total:,} articles in {elapsed:.1f}s ({total/elapsed:.0f} articles/sec)")

        # Train quantizer on sample if available
        if HAS_QUANTIZATION and self.quantizer and len(articles) >= 1000:
            print("  Training quantizer on sample data...")
            sample_vectors = np.array([a.embedding for a in articles[:10000]], dtype=np.float32)
            self.quantizer.train(sample_vectors)
            print("  Quantizer trained")

    def build_knowledge_graph(self, articles: List[Article], max_articles: int = 50000):
        """
        Build comprehensive knowledge graph from article entities.

        Creates a rich graph structure with:
        - Article nodes with full metadata
        - Entity nodes (Companies, People, Locations, Organizations)
        - Topic nodes for each category/topic
        - Source nodes for news sources
        - Multiple relationship types (MENTIONS, BELONGS_TO, PUBLISHED_BY, etc.)
        - Hyperedges for co-mentions (multiple entities in same article)
        """
        if not HAS_GRAPH or not self.graph:
            print("Knowledge graph not available")
            return

        print(f"\nBuilding comprehensive knowledge graph from {min(len(articles), max_articles):,} articles...")
        start_time = time.time()

        entity_nodes = {}
        topic_nodes = {}
        source_nodes = {}
        article_count = 0
        edge_count = 0
        hyperedge_count = 0

        # First pass: Create all topic and source nodes
        print("  Creating topic and source nodes...")
        for category, data in NEWS_CATEGORIES.items():
            # Create category node
            category_id = f"category_{category}"
            if category_id not in topic_nodes:
                category_node = NodeBuilder(category_id)\
                    .label("Category")\
                    .property("name", category)\
                    .property("weight", data["weight"])\
                    .build()
                self.graph.create_node(category_node)
                topic_nodes[category_id] = True

            # Create topic nodes under each category
            for topic in data["topics"]:
                topic_id = f"topic_{topic.replace(' ', '_').lower()}"
                if topic_id not in topic_nodes:
                    topic_node = NodeBuilder(topic_id)\
                        .label("Topic")\
                        .property("name", topic)\
                        .property("category", category)\
                        .build()
                    self.graph.create_node(topic_node)
                    topic_nodes[topic_id] = True

                    # Link topic to category
                    edge = EdgeBuilder(topic_id, category_id, "BELONGS_TO").build()
                    self.graph.create_edge(edge)
                    edge_count += 1

            # Create source nodes
            for source in data["sources"]:
                source_id = f"source_{source.replace(' ', '_').lower()}"
                if source_id not in source_nodes:
                    source_node = NodeBuilder(source_id)\
                        .label("Source")\
                        .property("name", source)\
                        .property("category", category)\
                        .build()
                    self.graph.create_node(source_node)
                    source_nodes[source_id] = True

        print(f"    Created {len(topic_nodes)} topic nodes, {len(source_nodes)} source nodes")

        # Second pass: Create entity nodes (all companies, people, etc.)
        print("  Creating entity nodes...")
        for entity_type, entities_list in ENTITIES.items():
            label = entity_type[:-1].title() if entity_type.endswith('s') else entity_type.title()
            # Map to correct labels
            if entity_type == "companies":
                label = "Company"
            elif entity_type == "people":
                label = "Person"
            elif entity_type == "locations":
                label = "Location"
            elif entity_type == "organizations":
                label = "Organization"

            for entity in entities_list:
                entity_id = f"entity_{entity.replace(' ', '_').lower()}"
                if entity_id not in entity_nodes:
                    entity_node = NodeBuilder(entity_id)\
                        .label("Entity")\
                        .label(label)\
                        .property("name", entity)\
                        .property("type", label)\
                        .build()
                    self.graph.create_node(entity_node)
                    entity_nodes[entity_id] = True

        print(f"    Created {len(entity_nodes)} entity nodes")

        # Create relationships between related entities
        print("  Creating entity relationships...")
        # Companies and their notable people
        company_people_links = [
            ("Apple", "Tim Cook"), ("Tesla", "Elon Musk"), ("Microsoft", "Satya Nadella"),
            ("Google", "Sundar Pichai"), ("Meta", "Mark Zuckerberg"), ("Amazon", "Andy Jassy"),
            ("Amazon", "Jeff Bezos"), ("OpenAI", "Sam Altman"), ("Anthropic", "Dario Amodei"),
            ("Nvidia", "Jensen Huang"), ("Intel", "Pat Gelsinger"), ("AMD", "Lisa Su"),
            ("JPMorgan", "Jamie Dimon")
        ]
        for company, person in company_people_links:
            company_id = f"entity_{company.replace(' ', '_').lower()}"
            person_id = f"entity_{person.replace(' ', '_').lower()}"
            if company_id in entity_nodes and person_id in entity_nodes:
                edge = EdgeBuilder(person_id, company_id, "LEADS")\
                    .property("role", "CEO/Leader")\
                    .build()
                self.graph.create_edge(edge)
                edge_count += 1

        # Location relationships (companies headquartered in)
        company_locations = [
            ("Apple", "Silicon Valley"), ("Google", "Silicon Valley"), ("Meta", "Silicon Valley"),
            ("Microsoft", "Seattle"), ("Amazon", "Seattle"), ("Tesla", "Austin"),
            ("OpenAI", "San Francisco"), ("Anthropic", "San Francisco"),
            ("JPMorgan", "New York"), ("Goldman Sachs", "New York")
        ]
        for company, location in company_locations:
            company_id = f"entity_{company.replace(' ', '_').lower()}"
            location_id = f"entity_{location.replace(' ', '_').lower()}"
            if company_id in entity_nodes and location_id in entity_nodes:
                edge = EdgeBuilder(company_id, location_id, "HEADQUARTERED_IN").build()
                self.graph.create_edge(edge)
                edge_count += 1

        print(f"    Created {edge_count} entity relationship edges")

        # Third pass: Process articles
        print("  Processing articles...")
        for article in articles[:max_articles]:
            # Create article node with rich metadata
            article_node = NodeBuilder(article.id)\
                .label("Article")\
                .property("headline", article.headline[:200])\
                .property("category", article.category)\
                .property("topic", article.topic)\
                .property("source", article.source)\
                .property("date", article.published_date.isoformat())\
                .property("year", article.published_date.year)\
                .property("month", article.published_date.month)\
                .property("sentiment", article.sentiment)\
                .property("word_count", article.word_count)\
                .build()

            self.graph.create_node(article_node)

            # Link article to topic
            topic_id = f"topic_{article.topic.replace(' ', '_').lower()}"
            if topic_id in topic_nodes:
                edge = EdgeBuilder(article.id, topic_id, "ABOUT")\
                    .property("relevance", 1.0)\
                    .build()
                self.graph.create_edge(edge)
                edge_count += 1

            # Link article to category
            category_id = f"category_{article.category}"
            if category_id in topic_nodes:
                edge = EdgeBuilder(article.id, category_id, "IN_CATEGORY").build()
                self.graph.create_edge(edge)
                edge_count += 1

            # Link article to source
            source_id = f"source_{article.source.replace(' ', '_').lower()}"
            if source_id in source_nodes:
                edge = EdgeBuilder(article.id, source_id, "PUBLISHED_BY")\
                    .property("date", article.published_date.isoformat())\
                    .build()
                self.graph.create_edge(edge)
                edge_count += 1

            # Create MENTIONS relationships for each entity
            for entity in article.entities:
                entity_id = f"entity_{entity.replace(' ', '_').lower()}"
                if entity_id in entity_nodes:
                    edge = EdgeBuilder(article.id, entity_id, "MENTIONS")\
                        .property("context", article.topic)\
                        .property("sentiment", article.sentiment)\
                        .build()
                    self.graph.create_edge(edge)
                    edge_count += 1

            # Create hyperedge for co-mentions (when article mentions 2+ entities)
            if len(article.entities) >= 2:
                entity_ids = [f"entity_{e.replace(' ', '_').lower()}" for e in article.entities]
                valid_entity_ids = [eid for eid in entity_ids if eid in entity_nodes]

                if len(valid_entity_ids) >= 2:
                    # Create a hyperedge connecting all co-mentioned entities
                    hyperedge = HyperedgeBuilder(valid_entity_ids, "CO_MENTIONED")\
                        .property("article_id", article.id)\
                        .property("topic", article.topic)\
                        .property("date", article.published_date.isoformat())\
                        .build()
                    try:
                        self.graph.create_hyperedge(hyperedge)
                        hyperedge_count += 1
                    except:
                        pass  # Hyperedge may already exist

            # Link to related topics
            for related_topic in article.related_topics[:2]:  # Limit to avoid explosion
                related_topic_id = f"topic_{related_topic.replace(' ', '_').lower()}"
                if related_topic_id in topic_nodes:
                    edge = EdgeBuilder(article.id, related_topic_id, "RELATED_TO")\
                        .property("strength", 0.5)\
                        .build()
                    try:
                        self.graph.create_edge(edge)
                        edge_count += 1
                    except:
                        pass

            article_count += 1
            if article_count % 5000 == 0:
                print(f"\r    Processed {article_count:,} articles, {edge_count:,} edges...", end="")

        # Save graph
        self.graph.save()

        elapsed = time.time() - start_time
        total_nodes = len(entity_nodes) + len(topic_nodes) + len(source_nodes) + article_count

        print(f"\n  Knowledge Graph Statistics:")
        print(f"    - Total nodes: {total_nodes:,}")
        print(f"      - Articles: {article_count:,}")
        print(f"      - Entities: {len(entity_nodes):,}")
        print(f"      - Topics: {len(topic_nodes):,}")
        print(f"      - Sources: {len(source_nodes):,}")
        print(f"    - Total edges: {edge_count:,}")
        print(f"    - Hyperedges (co-mentions): {hyperedge_count:,}")
        print(f"    - Build time: {elapsed:.1f}s")

    def semantic_search(self, query_text: str, query_embedding: np.ndarray,
                       k: int = 10, filters: Optional[Dict] = None) -> List:
        """Perform semantic search with optional filters."""
        filter_obj = None
        if filters:
            filter_conditions = []
            for key, value in filters.items():
                if isinstance(value, list):
                    filter_conditions.append(Filter.in_(key, value))
                elif isinstance(value, tuple) and len(value) == 2:
                    # Range filter (min, max)
                    filter_conditions.append(Filter.gte(key, value[0]))
                    filter_conditions.append(Filter.lte(key, value[1]))
                else:
                    filter_conditions.append(Filter.eq(key, value))

            if len(filter_conditions) == 1:
                filter_obj = filter_conditions[0]
            elif len(filter_conditions) > 1:
                filter_obj = Filter.and_(filter_conditions)

        results = self.collection.search(query_embedding, k=k, filter=filter_obj)
        return results

    def batch_search(self, query_embeddings: np.ndarray, k: int = 10) -> List[List]:
        """Perform batch search for multiple queries."""
        return self.collection.search_batch(query_embeddings, k=k)

    def get_statistics(self) -> Dict:
        """Get database statistics."""
        stats = {
            "total_articles": self.collection.count(),
            "embedding_dimensions": self.config.embedding_dim,
            "hnsw_m": self.config.hnsw_m,
            "hnsw_ef_search": self.config.hnsw_ef_search,
        }

        # Estimate memory usage
        num_vectors = stats["total_articles"]
        vector_bytes = num_vectors * self.config.embedding_dim * 4  # float32
        stats["estimated_vector_memory_mb"] = vector_bytes / (1024 * 1024)

        if HAS_QUANTIZATION and self.quantizer:
            quantized_bytes = num_vectors * self.config.embedding_dim  # uint8
            stats["quantized_memory_mb"] = quantized_bytes / (1024 * 1024)
            stats["compression_ratio"] = vector_bytes / quantized_bytes

        return stats


# ================================================================================
# DEMO & BENCHMARKING
# ================================================================================

class NewsIntelligenceDemo:
    """Demonstrates news intelligence platform capabilities."""

    def __init__(self, config: Config):
        self.config = config
        self.db = NewsIntelligenceDB(config)
        self.generator = NewsDataGenerator(config)
        self.articles = []

    def setup(self):
        """Set up the demo environment."""
        print("=" * 80)
        print("NEWS INTELLIGENCE PLATFORM DEMO")
        print("=" * 80)
        print(f"\nConfiguration:")
        print(f"  - Articles to generate: {self.config.num_articles:,}")
        print(f"  - Embedding dimensions: {self.config.embedding_dim}")
        print(f"  - Database path: {self.config.db_path}")
        print(f"  - HNSW M parameter: {self.config.hnsw_m}")
        print(f"  - Batch size: {self.config.batch_size:,}")

        # Initialize database
        self.db.initialize()

        # Check if we need to generate data
        existing_count = self.db.collection.count()
        if existing_count >= self.config.num_articles:
            print(f"\nDatabase already contains {existing_count:,} articles. Skipping generation.")
            return

        # Generate articles
        self.articles = self.generator.generate_all()

        # Ingest into database
        self.db.ingest_articles(self.articles)

        # Build knowledge graph (limited for demo)
        if HAS_GRAPH:
            self.db.build_knowledge_graph(self.articles, max_articles=10000)

    def run_search_demo(self):
        """Run various search demonstrations."""
        print("\n" + "=" * 80)
        print("SEARCH DEMONSTRATIONS")
        print("=" * 80)

        # Generate sample queries based on topics
        print("\n--- 1. Basic Semantic Search ---")
        sample_topics = [
            "artificial intelligence",
            "climate change",
            "stock market",
            "space exploration",
            "healthcare reform"
        ]

        for topic in sample_topics:
            # Get topic embedding
            query_embedding = self.generator.topic_embeddings.get(
                topic,
                self.generator.rng.standard_normal(self.config.embedding_dim).astype(np.float32)
            )

            results = self.db.semantic_search(topic, query_embedding, k=5)

            print(f"\n  Query: '{topic}'")
            print(f"  Results:")
            for i, r in enumerate(results[:3], 1):
                headline = r.metadata.get("headline", "N/A")[:60]
                print(f"    {i}. [{r.score:.4f}] {headline}...")

        # Filtered search
        print("\n--- 2. Filtered Search (Category + Time Range) ---")
        query_embedding = self.generator.topic_embeddings.get(
            "artificial intelligence",
            self.generator.rng.standard_normal(self.config.embedding_dim).astype(np.float32)
        )

        results = self.db.semantic_search(
            "AI technology",
            query_embedding,
            k=5,
            filters={"category": "technology"}
        )

        print(f"\n  Query: 'AI technology' (filtered to 'technology' category)")
        print(f"  Results:")
        for i, r in enumerate(results[:3], 1):
            headline = r.metadata.get("headline", "N/A")[:50]
            category = r.metadata.get("category", "N/A")
            print(f"    {i}. [{r.score:.4f}] [{category}] {headline}...")

        # Sentiment-based search
        print("\n--- 3. Sentiment-Based Search ---")
        for sentiment in ["positive", "negative"]:
            results = self.db.semantic_search(
                "market news",
                query_embedding,
                k=3,
                filters={"sentiment": sentiment, "category": "business"}
            )

            print(f"\n  Query: 'market news' (sentiment: {sentiment})")
            for i, r in enumerate(results[:2], 1):
                headline = r.metadata.get("headline", "N/A")[:50]
                print(f"    {i}. [{r.score:.4f}] {headline}...")

        # Source-based search
        print("\n--- 4. Source-Based Search ---")
        results = self.db.semantic_search(
            "technology trends",
            query_embedding,
            k=5,
            filters={"source": ["TechCrunch", "Wired", "The Verge"]}
        )

        print(f"\n  Query: 'technology trends' (sources: TechCrunch, Wired, The Verge)")
        for i, r in enumerate(results[:3], 1):
            headline = r.metadata.get("headline", "N/A")[:45]
            source = r.metadata.get("source", "N/A")
            print(f"    {i}. [{r.score:.4f}] [{source}] {headline}...")

    def run_benchmark(self):
        """Run performance benchmarks."""
        print("\n" + "=" * 80)
        print("PERFORMANCE BENCHMARKS")
        print("=" * 80)

        num_queries = self.config.demo_queries

        # Generate random query embeddings
        query_embeddings = self.generator.rng.standard_normal(
            (num_queries, self.config.embedding_dim)
        ).astype(np.float32)

        # Normalize
        norms = np.linalg.norm(query_embeddings, axis=1, keepdims=True)
        query_embeddings = query_embeddings / norms

        # Single query benchmark
        print(f"\n--- Single Query Latency (k=10) ---")
        latencies = []
        for i in range(min(100, num_queries)):
            start = time.perf_counter()
            _ = self.db.collection.search(query_embeddings[i], k=10)
            latencies.append((time.perf_counter() - start) * 1000)

        print(f"  Queries: {len(latencies)}")
        print(f"  Mean latency: {np.mean(latencies):.2f} ms")
        print(f"  P50 latency: {np.percentile(latencies, 50):.2f} ms")
        print(f"  P95 latency: {np.percentile(latencies, 95):.2f} ms")
        print(f"  P99 latency: {np.percentile(latencies, 99):.2f} ms")
        print(f"  Throughput: {1000 / np.mean(latencies):.0f} QPS")

        # Batch query benchmark
        print(f"\n--- Batch Query Latency ({num_queries} queries, k=10) ---")
        start = time.perf_counter()
        _ = self.db.batch_search(query_embeddings, k=10)
        batch_time = (time.perf_counter() - start) * 1000

        print(f"  Total time: {batch_time:.2f} ms")
        print(f"  Per-query time: {batch_time / num_queries:.2f} ms")
        print(f"  Throughput: {num_queries * 1000 / batch_time:.0f} QPS")

        # Filtered search benchmark
        print(f"\n--- Filtered Search Latency (category filter) ---")
        filtered_latencies = []
        for i in range(min(50, num_queries)):
            start = time.perf_counter()
            _ = self.db.semantic_search(
                "test",
                query_embeddings[i],
                k=10,
                filters={"category": "technology"}
            )
            filtered_latencies.append((time.perf_counter() - start) * 1000)

        print(f"  Mean latency: {np.mean(filtered_latencies):.2f} ms")
        print(f"  P95 latency: {np.percentile(filtered_latencies, 95):.2f} ms")

        # Print database statistics
        print(f"\n--- Database Statistics ---")
        stats = self.db.get_statistics()
        print(f"  Total articles: {stats['total_articles']:,}")
        print(f"  Embedding dimensions: {stats['embedding_dimensions']}")
        print(f"  Estimated vector memory: {stats['estimated_vector_memory_mb']:.1f} MB")
        if 'quantized_memory_mb' in stats:
            print(f"  Quantized memory: {stats['quantized_memory_mb']:.1f} MB")
            print(f"  Compression ratio: {stats['compression_ratio']:.1f}x")

    def run_graph_demo(self):
        """Demonstrate comprehensive knowledge graph capabilities."""
        if not HAS_GRAPH or not self.db.graph:
            print("\nGraph module not available, skipping demo.")
            return

        print("\n" + "=" * 80)
        print("KNOWLEDGE GRAPH DEMONSTRATIONS")
        print("=" * 80)

        graph = self.db.graph

        # 1. Graph Statistics Overview
        print("\n--- 1. Knowledge Graph Statistics ---")
        stats = graph.stats()
        print(f"  Nodes: {stats['nodes']:,}")
        print(f"  Edges: {stats['edges']:,}")
        print(f"  Hyperedges: {stats['hyperedges']:,}")
        print(f"  Node labels: {', '.join(stats['labels'][:10])}...")
        print(f"  Edge types: {', '.join(stats['edge_types'][:10])}...")

        # 2. Entity Queries - Find all companies
        print("\n--- 2. Entity Queries ---")
        print("  Finding all companies in the graph...")
        companies = graph.find_nodes(label="Company")
        print(f"  Found {len(companies)} companies:")
        for company in companies[:5]:
            print(f"    - {company.properties.get('name', company.id)}")

        print("\n  Finding all people in the graph...")
        people = graph.find_nodes(label="Person")
        print(f"  Found {len(people)} people:")
        for person in people[:5]:
            print(f"    - {person.properties.get('name', person.id)}")

        # 3. Relationship Traversal - Who leads which company?
        print("\n--- 3. Relationship Traversal ---")
        print("  Query: Who leads which company? (Person)-[:LEADS]->(Company)")

        leaders_found = 0
        for person in people:
            neighbors = graph.neighbors(person.id, direction="out", edge_type="LEADS")
            if neighbors:
                person_name = person.properties.get('name', person.id)
                for company in neighbors:
                    company_name = company.properties.get('name', company.id)
                    print(f"    {person_name} leads {company_name}")
                    leaders_found += 1
                    if leaders_found >= 5:
                        break
            if leaders_found >= 5:
                break

        # 4. Multi-hop Traversal - Find articles about companies in Silicon Valley
        print("\n--- 4. Multi-hop Graph Traversal ---")
        print("  Query: Find articles mentioning companies headquartered in Silicon Valley")

        silicon_valley_id = "entity_silicon_valley"
        sv_node = graph.get_node(silicon_valley_id)

        if sv_node:
            # Find companies in Silicon Valley (incoming HEADQUARTERED_IN edges)
            sv_companies = graph.neighbors(silicon_valley_id, direction="in", edge_type="HEADQUARTERED_IN")
            print(f"  Companies in Silicon Valley: {[c.properties.get('name') for c in sv_companies[:5]]}")

            # For each company, find articles that mention them
            article_count = 0
            for company in sv_companies[:3]:
                company_name = company.properties.get('name', company.id)
                articles = graph.neighbors(company.id, direction="in", edge_type="MENTIONS")
                print(f"\n    {company_name}: {len(articles)} articles mention this company")
                for article in articles[:2]:
                    headline = article.properties.get('headline', '')[:60]
                    print(f"      - {headline}...")
                article_count += len(articles)

            print(f"\n  Total articles found: {article_count}")

        # 5. Cypher-like Queries
        print("\n--- 5. Cypher-like Queries ---")

        # Query: Find all topics
        print("  Query: MATCH (t:Topic) RETURN t.name, t.category")
        try:
            results = graph.query("MATCH (t:Topic) RETURN t.name, t.category")
            print(f"  Found {len(results)} topics. First 5:")
            for row in list(results)[:5]:
                print(f"    - {row.get('t.name', 'N/A')} ({row.get('t.category', 'N/A')})")
        except Exception as e:
            print(f"  Query error: {e}")

        # Query: Find articles by category
        print("\n  Query: MATCH (a:Article) WHERE a.category = 'technology' RETURN a.headline")
        try:
            results = graph.query("MATCH (a:Article) WHERE a.category = technology RETURN a.headline")
            print(f"  Found {len(results)} technology articles. First 3:")
            for row in list(results)[:3]:
                headline = row.get('a.headline', 'N/A')[:60]
                print(f"    - {headline}...")
        except Exception as e:
            print(f"  Query error: {e}")

        # 6. Path Finding
        print("\n--- 6. Path Finding ---")

        # Find path from Tim Cook to any article
        tim_cook_id = "entity_tim_cook"
        apple_id = "entity_apple"

        if graph.get_node(tim_cook_id) and graph.get_node(apple_id):
            print(f"  Finding shortest path from Tim Cook to Apple...")
            path = graph.shortest_path(tim_cook_id, apple_id, max_depth=5)
            if path:
                path_names = []
                for node in path:
                    name = node.properties.get('name') or node.properties.get('headline', node.id)[:30]
                    path_names.append(name)
                print(f"    Path: {' -> '.join(path_names)}")
            else:
                print("    No path found")

        # 7. Graph Traversal - Deep exploration
        print("\n--- 7. Deep Graph Traversal ---")
        print("  Exploring 3-hop neighborhood from 'artificial intelligence' topic...")

        ai_topic_id = "topic_artificial_intelligence"
        if graph.get_node(ai_topic_id):
            paths = graph.traverse(ai_topic_id, max_depth=2, direction="in")
            print(f"  Found {len(paths)} paths (showing first 5):")
            for path in paths[:5]:
                path_str = " -> ".join([
                    n.properties.get('name') or n.properties.get('headline', n.id)[:25]
                    for n in path
                ])
                print(f"    {path_str}")

        # 8. Hyperedge Queries - Co-mention analysis
        print("\n--- 8. Hyperedge Analysis (Co-mentions) ---")
        print("  Entities frequently mentioned together in articles:")

        # Get hyperedges for a specific entity
        elon_musk_id = "entity_elon_musk"
        if graph.get_node(elon_musk_id):
            hyperedges = graph.get_hyperedges_by_node(elon_musk_id)
            print(f"\n  Elon Musk is co-mentioned in {len(hyperedges)} article contexts")

            if hyperedges:
                print("  Sample co-mentions:")
                for he in hyperedges[:3]:
                    co_entities = []
                    for node_id in he.nodes:
                        if node_id != elon_musk_id:
                            node = graph.get_node(node_id)
                            if node:
                                co_entities.append(node.properties.get('name', node_id))
                    topic = he.properties.get('topic', 'N/A')
                    print(f"    - Co-mentioned with: {', '.join(co_entities)} (topic: {topic})")

        # 9. Source Analysis
        print("\n--- 9. Source/Publisher Analysis ---")
        print("  Articles by news source:")

        sources = graph.find_nodes(label="Source")
        for source in sources[:5]:
            source_name = source.properties.get('name', source.id)
            articles = graph.neighbors(source.id, direction="in", edge_type="PUBLISHED_BY")
            print(f"    - {source_name}: {len(articles)} articles")

        # 10. Category Distribution
        print("\n--- 10. Category Distribution ---")
        categories = graph.find_nodes(label="Category")
        for category in categories:
            cat_name = category.properties.get('name', category.id)
            # Count articles in this category
            articles = graph.neighbors(category.id, direction="in", edge_type="IN_CATEGORY")
            # Count topics in this category
            topics = graph.neighbors(category.id, direction="in", edge_type="BELONGS_TO")
            print(f"    - {cat_name}: {len(articles)} articles, {len(topics)} topics")

    def run_quantization_demo(self):
        """Demonstrate quantization capabilities."""
        if not HAS_QUANTIZATION:
            print("\nQuantization module not available, skipping demo.")
            return

        print("\n" + "=" * 80)
        print("QUANTIZATION DEMONSTRATION")
        print("=" * 80)

        # Generate sample vectors
        sample_size = 10000
        sample_vectors = self.generator.rng.standard_normal(
            (sample_size, self.config.embedding_dim)
        ).astype(np.float32)
        norms = np.linalg.norm(sample_vectors, axis=1, keepdims=True)
        sample_vectors = sample_vectors / norms

        # Scalar quantization
        print(f"\n--- Scalar Quantization (8-bit) ---")
        sq = ScalarQuantizer(self.config.embedding_dim)
        sq.train(sample_vectors)

        start = time.perf_counter()
        quantized = sq.encode(sample_vectors)
        encode_time = (time.perf_counter() - start) * 1000

        original_size = sample_vectors.nbytes / (1024 * 1024)
        quantized_size = quantized.nbytes / (1024 * 1024)

        print(f"  Original size: {original_size:.2f} MB")
        print(f"  Quantized size: {quantized_size:.2f} MB")
        print(f"  Compression ratio: {original_size / quantized_size:.1f}x")
        print(f"  Encode time: {encode_time:.2f} ms ({sample_size * 1000 / encode_time:.0f} vectors/sec)")

        # Test recall
        query = sample_vectors[0:1]
        reconstructed = sq.decode(quantized)

        # Calculate approximate recall
        original_distances = np.linalg.norm(sample_vectors - query, axis=1)
        reconstructed_distances = np.linalg.norm(reconstructed - query, axis=1)

        k = 100
        original_neighbors = set(np.argsort(original_distances)[:k])
        reconstructed_neighbors = set(np.argsort(reconstructed_distances)[:k])
        recall = len(original_neighbors & reconstructed_neighbors) / k

        print(f"  Approximate recall@{k}: {recall:.1%}")

        # Binary quantization
        print(f"\n--- Binary Quantization (1-bit) ---")
        try:
            bq = BinaryQuantizer(self.config.embedding_dim)
            bq.train(sample_vectors)

            binary_codes = bq.encode(sample_vectors)
            binary_size = binary_codes.nbytes / (1024 * 1024)

            print(f"  Original size: {original_size:.2f} MB")
            print(f"  Binary size: {binary_size:.2f} MB")
            print(f"  Compression ratio: {original_size / binary_size:.1f}x")
        except Exception as e:
            print(f"  Binary quantization unavailable: {e}")

    def run_use_case_scenarios(self):
        """Run realistic use case scenarios."""
        print("\n" + "=" * 80)
        print("REAL-WORLD USE CASE SCENARIOS")
        print("=" * 80)

        # Scenario 1: Breaking News Alert
        print("\n--- Scenario 1: Breaking News Alert System ---")
        print("  Use case: Find all related articles when breaking news occurs")

        # Simulate breaking news about AI
        breaking_topic = "artificial intelligence"
        query_embedding = self.generator.topic_embeddings.get(breaking_topic)

        print(f"\n  Breaking news topic: '{breaking_topic}'")
        print("  Finding related coverage across all sources...")

        results = self.db.semantic_search(
            breaking_topic,
            query_embedding,
            k=10
        )

        # Group by source
        by_source = defaultdict(list)
        for r in results:
            source = r.metadata.get("source", "Unknown")
            by_source[source].append(r)

        print(f"\n  Coverage by source:")
        for source, articles in by_source.items():
            print(f"    - {source}: {len(articles)} articles")

        # Scenario 2: Competitive Intelligence
        print("\n--- Scenario 2: Competitive Intelligence ---")
        print("  Use case: Track mentions of specific companies")

        companies = ["Apple", "Google", "Microsoft"]
        print(f"\n  Tracking companies: {', '.join(companies)}")

        for company in companies:
            # Search for company-related content
            company_embedding = self.generator.rng.standard_normal(
                self.config.embedding_dim
            ).astype(np.float32)
            company_embedding = company_embedding / np.linalg.norm(company_embedding)

            results = self.db.semantic_search(
                company,
                company_embedding,
                k=5,
                filters={"category": ["technology", "business"]}
            )

            sentiments = [r.metadata.get("sentiment", "neutral") for r in results]
            sentiment_counts = {s: sentiments.count(s) for s in set(sentiments)}

            print(f"\n  {company}:")
            print(f"    - Articles found: {len(results)}")
            print(f"    - Sentiment distribution: {sentiment_counts}")

        # Scenario 3: Research Assistant
        print("\n--- Scenario 3: Research Assistant ---")
        print("  Use case: Find comprehensive coverage on a research topic")

        research_topic = "climate change"
        query_embedding = self.generator.topic_embeddings.get(research_topic)

        print(f"\n  Research topic: '{research_topic}'")

        # Get articles from different categories
        categories = ["science", "politics", "business", "world"]
        print(f"  Searching across categories: {', '.join(categories)}")

        for category in categories:
            results = self.db.semantic_search(
                research_topic,
                query_embedding,
                k=3,
                filters={"category": category}
            )

            if results:
                print(f"\n  [{category.upper()}]:")
                for i, r in enumerate(results[:2], 1):
                    headline = r.metadata.get("headline", "N/A")[:50]
                    source = r.metadata.get("source", "N/A")
                    print(f"    {i}. {headline}... ({source})")

        # Scenario 4: Trend Analysis
        print("\n--- Scenario 4: Trend Analysis ---")
        print("  Use case: Analyze topic trends over time")

        trend_topic = "cryptocurrency"
        query_embedding = self.generator.topic_embeddings.get(
            trend_topic,
            self.generator.rng.standard_normal(self.config.embedding_dim).astype(np.float32)
        )

        print(f"\n  Analyzing trends for: '{trend_topic}'")

        # Search with different time filters (simulated via date in metadata)
        results = self.db.semantic_search(
            trend_topic,
            query_embedding,
            k=50
        )

        # Analyze sentiment trends
        by_year = defaultdict(list)
        for r in results:
            date_str = r.metadata.get("published_date", "")
            if date_str:
                try:
                    year = datetime.fromisoformat(date_str).year
                    by_year[year].append(r.metadata.get("sentiment", "neutral"))
                except:
                    pass

        print(f"\n  Sentiment trends by year:")
        for year in sorted(by_year.keys()):
            sentiments = by_year[year]
            pos = sentiments.count("positive")
            neg = sentiments.count("negative")
            total = len(sentiments)
            print(f"    {year}: {total} articles (positive: {pos}, negative: {neg})")

        # Scenario 5: Entity Network Analysis (Graph-based)
        if HAS_GRAPH and self.db.graph:
            print("\n--- Scenario 5: Entity Network Analysis (Graph-Powered) ---")
            print("  Use case: Map relationships between entities mentioned in news")

            graph = self.db.graph

            # Find most connected entities
            print("\n  Top connected entities (by article mentions):")
            entities = graph.find_nodes(label="Entity")

            entity_connections = []
            for entity in entities[:50]:  # Check first 50 for speed
                mentions = graph.neighbors(entity.id, direction="in", edge_type="MENTIONS")
                entity_name = entity.properties.get('name', entity.id)
                entity_type = entity.properties.get('type', 'Unknown')
                entity_connections.append((entity_name, entity_type, len(mentions)))

            # Sort by connection count
            entity_connections.sort(key=lambda x: x[2], reverse=True)

            for name, etype, count in entity_connections[:10]:
                print(f"    - {name} ({etype}): {count} article mentions")

            # Scenario 6: Corporate Ecosystem Mapping
            print("\n--- Scenario 6: Corporate Ecosystem Mapping ---")
            print("  Use case: Map the ecosystem around a company")

            target_company = "Google"
            company_id = f"entity_{target_company.lower()}"

            if graph.get_node(company_id):
                print(f"\n  Ecosystem map for {target_company}:")

                # Find leaders
                leaders = graph.neighbors(company_id, direction="in", edge_type="LEADS")
                if leaders:
                    print(f"    Leaders: {[l.properties.get('name') for l in leaders]}")

                # Find headquarters
                locations = graph.neighbors(company_id, direction="out", edge_type="HEADQUARTERED_IN")
                if locations:
                    print(f"    Headquarters: {[l.properties.get('name') for l in locations]}")

                # Find co-mentioned entities via hyperedges
                hyperedges = graph.get_hyperedges_by_node(company_id)
                co_mentioned_entities = set()
                for he in hyperedges[:50]:  # Limit for speed
                    for node_id in he.nodes:
                        if node_id != company_id:
                            node = graph.get_node(node_id)
                            if node:
                                co_mentioned_entities.add(node.properties.get('name', node_id))

                if co_mentioned_entities:
                    print(f"    Frequently co-mentioned with: {list(co_mentioned_entities)[:10]}")

                # Find articles mentioning this company
                articles = graph.neighbors(company_id, direction="in", edge_type="MENTIONS")
                if articles:
                    # Analyze sentiment of coverage
                    sentiments = [a.properties.get('sentiment', 'neutral') for a in articles]
                    sentiment_dist = {s: sentiments.count(s) for s in set(sentiments)}
                    print(f"    Coverage sentiment: {sentiment_dist}")

                    # Topics covered
                    topics = set()
                    for article in articles[:100]:
                        topics.add(article.properties.get('topic', 'unknown'))
                    print(f"    Topics covered: {list(topics)[:8]}")

            # Scenario 7: Graph + Vector Semantic Search
            print("\n--- Scenario 7: Graph-Augmented Semantic Search ---")
            print("  Use case: Find articles via semantic search, then explore graph context")

            # First, find semantically similar articles
            query_topic = "machine learning"
            query_embedding = self.generator.topic_embeddings.get(
                query_topic,
                self.generator.rng.standard_normal(self.config.embedding_dim).astype(np.float32)
            )

            print(f"\n  Step 1: Vector search for '{query_topic}'")
            results = self.db.semantic_search(query_topic, query_embedding, k=5)

            for i, r in enumerate(results[:3], 1):
                article_id = r.id
                headline = r.metadata.get("headline", "N/A")[:50]
                print(f"\n    Article {i}: {headline}...")

                # Use graph to find related entities
                article_node = graph.get_node(article_id)
                if article_node:
                    # Find entities mentioned in this article
                    entities = graph.neighbors(article_id, direction="out", edge_type="MENTIONS")
                    if entities:
                        entity_names = [e.properties.get('name') for e in entities[:5]]
                        print(f"      Entities mentioned: {entity_names}")

                    # Find the source
                    sources = graph.neighbors(article_id, direction="out", edge_type="PUBLISHED_BY")
                    if sources:
                        source_name = sources[0].properties.get('name')
                        print(f"      Published by: {source_name}")

                    # Find related topics
                    topics = graph.neighbors(article_id, direction="out", edge_type="ABOUT")
                    if topics:
                        topic_names = [t.properties.get('name') for t in topics]
                        print(f"      Topics: {topic_names}")

    def run_all(self):
        """Run the complete demonstration."""
        try:
            self.setup()
            self.run_search_demo()
            self.run_benchmark()
            self.run_graph_demo()
            self.run_quantization_demo()
            self.run_use_case_scenarios()

            print("\n" + "=" * 80)
            print("DEMO COMPLETE")
            print("=" * 80)

            stats = self.db.get_statistics()
            print(f"\nFinal Statistics:")
            print(f"  - Total articles indexed: {stats['total_articles']:,}")
            print(f"  - Vector memory usage: {stats['estimated_vector_memory_mb']:.1f} MB")
            print(f"  - Database location: {self.config.db_path}")

            print("\nThe database is now ready for queries. You can:")
            print("  1. Run this script again to query the existing database")
            print("  2. Import NewsIntelligenceDB in your own scripts")
            print("  3. Start the REST API server with: python server.py")

        except Exception as e:
            print(f"\nError during demo: {e}")
            import traceback
            traceback.print_exc()


# ================================================================================
# MAIN ENTRY POINT
# ================================================================================

def main():
    """Main entry point for the news intelligence demo."""
    import argparse

    parser = argparse.ArgumentParser(
        description="News Intelligence Platform - PyVectorDB Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default settings (100K articles)
  python news_intelligence_demo.py

  # Run with custom article count
  python news_intelligence_demo.py --articles 500000

  # Run with custom database path
  python news_intelligence_demo.py --db-path ./my_news_db

  # Quick demo with fewer articles
  python news_intelligence_demo.py --articles 10000 --quick
        """
    )

    parser.add_argument(
        "--articles", "-n",
        type=int,
        default=100_000,
        help="Number of articles to generate (default: 100,000)"
    )

    parser.add_argument(
        "--db-path",
        type=str,
        default="./news_intelligence_db",
        help="Path to store the database (default: ./news_intelligence_db)"
    )

    parser.add_argument(
        "--dimensions", "-d",
        type=int,
        default=384,
        help="Embedding dimensions (default: 384)"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=5000,
        help="Batch size for insertions (default: 5000)"
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel workers (default: 8)"
    )

    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick demo with reduced benchmarks"
    )

    parser.add_argument(
        "--benchmark-only",
        action="store_true",
        help="Only run benchmarks on existing database"
    )

    args = parser.parse_args()

    # Create configuration
    config = Config(
        num_articles=args.articles,
        embedding_dim=args.dimensions,
        db_path=args.db_path,
        batch_size=args.batch_size,
        num_workers=args.workers,
        demo_queries=50 if args.quick else 100
    )

    # Run demo
    demo = NewsIntelligenceDemo(config)

    if args.benchmark_only:
        demo.db.initialize()
        demo.run_benchmark()
    else:
        demo.run_all()


if __name__ == "__main__":
    main()
