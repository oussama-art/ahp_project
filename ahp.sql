--
-- PostgreSQL database dump
--

-- Dumped from database version 16.2
-- Dumped by pg_dump version 16.2

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

SET default_tablespace = '';

SET default_table_access_method = heap;

--
-- Name: criteria; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.criteria (
    id integer NOT NULL,
    criterion_name text,
    subcriteria_data jsonb,
    weights jsonb,
    consistent boolean,
    consistency_ratio numeric,
    ranking jsonb,
    subcriteria_comparisons jsonb,
    normalized_subcriteria_weights jsonb
);


ALTER TABLE public.criteria OWNER TO postgres;

--
-- Name: criteria_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.criteria_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.criteria_id_seq OWNER TO postgres;

--
-- Name: criteria_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.criteria_id_seq OWNED BY public.criteria.id;


--
-- Name: product; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.product (
    id integer NOT NULL,
    criteria jsonb,
    comparisons jsonb,
    weights jsonb,
    consistent boolean,
    consistency_ratio numeric,
    ranking jsonb,
    created_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    topsis_ranking jsonb,
    ideal_solution jsonb,
    negative_ideal_solution jsonb,
    relative_closeness jsonb
);


ALTER TABLE public.product OWNER TO postgres;

--
-- Name: product_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.product_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.product_id_seq OWNER TO postgres;

--
-- Name: product_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.product_id_seq OWNED BY public.product.id;


--
-- Name: users; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.users (
    id integer NOT NULL,
    fullname character varying(100) NOT NULL,
    username character varying(50) NOT NULL,
    password character varying(255) NOT NULL,
    email character varying(50) NOT NULL
);


ALTER TABLE public.users OWNER TO postgres;

--
-- Name: users_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.users_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.users_id_seq OWNER TO postgres;

--
-- Name: users_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.users_id_seq OWNED BY public.users.id;


--
-- Name: criteria id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.criteria ALTER COLUMN id SET DEFAULT nextval('public.criteria_id_seq'::regclass);


--
-- Name: product id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.product ALTER COLUMN id SET DEFAULT nextval('public.product_id_seq'::regclass);


--
-- Name: users id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.users ALTER COLUMN id SET DEFAULT nextval('public.users_id_seq'::regclass);


--
-- Data for Name: criteria; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.criteria (id, criterion_name, subcriteria_data, weights, consistent, consistency_ratio, ranking, subcriteria_comparisons, normalized_subcriteria_weights) FROM stdin;
582	test1	[{"name": "a1", "comparisons": {"c1": 1.0, "c2": 2.0, "c3": 3.0}}, {"name": "a2", "comparisons": {"c1": 0.5, "c2": 1.0, "c3": 4.0}}, {"name": "a3", "comparisons": {"c1": 0.3333333333333333, "c2": 0.25, "c3": 1.0}}]	[0.517133618631918, 0.3585604248241821, 0.12430595654389984]	t	0.048146131185257	[1, 2, 3]	\N	[0.5171336186319181, 0.35856042482418216, 0.12430595654389985]
583	test2	[{"name": "b1", "comparisons": {"c1": 1.0, "c2": 5.0, "c3": 3.0, "c4": 7.0}}, {"name": "b2", "comparisons": {"c1": 0.2, "c2": 1.0, "c3": 4.0, "c4": 2.0}}, {"name": "b3", "comparisons": {"c1": 0.3333333333333333, "c2": 0.25, "c3": 1.0, "c4": 5.0}}, {"name": "b4", "comparisons": {"c1": 0.14285714285714285, "c2": 0.5, "c3": 0.2, "c4": 1.0}}]	[0.5574026990750067, 0.22991618489252388, 0.15107691198427015, 0.06160420404819938]	f	0.17450029102444542	[1, 2, 3, 4]	\N	[0.5574026990750066, 0.22991618489252383, 0.15107691198427012, 0.061604204048199364]
584	test3	[{"name": "s1", "comparisons": {"c1": 1.0, "c2": 2.0, "c3": 6.0, "c4": 3.0}}, {"name": "s2", "comparisons": {"c1": 0.5, "c2": 1.0, "c3": 3.0, "c4": 2.0}}, {"name": "s3", "comparisons": {"c1": 0.16666666666666666, "c2": 0.3333333333333333, "c3": 1.0, "c4": 5.0}}, {"name": "s4", "comparisons": {"c1": 0.3333333333333333, "c2": 0.5, "c3": 0.2, "c4": 1.0}}]	[0.48920414875917173, 0.25417849879136223, 0.16772184449918304, 0.08889550795028295]	f	0.1724119921765265	[1, 2, 3, 4]	\N	[0.48920414875917173, 0.25417849879136223, 0.16772184449918304, 0.08889550795028295]
585	test1	[{"name": "a1", "comparisons": {"c1": 1.0, "c2": 2.0, "c3": 3.0}}, {"name": "a2", "comparisons": {"c1": 0.5, "c2": 1.0, "c3": 4.0}}, {"name": "a3", "comparisons": {"c1": 0.3333333333333333, "c2": 0.25, "c3": 1.0}}]	[0.517133618631918, 0.3585604248241821, 0.12430595654389984]	t	0.048146131185257	[1, 2, 3]	\N	[0.5171336186319181, 0.35856042482418216, 0.12430595654389985]
586	test2	[{"name": "b1", "comparisons": {"c1": 1.0, "c2": 5.0, "c3": 3.0, "c4": 7.0}}, {"name": "b2", "comparisons": {"c1": 0.2, "c2": 1.0, "c3": 4.0, "c4": 2.0}}, {"name": "b3", "comparisons": {"c1": 0.3333333333333333, "c2": 0.25, "c3": 1.0, "c4": 5.0}}, {"name": "b4", "comparisons": {"c1": 0.14285714285714285, "c2": 0.5, "c3": 0.2, "c4": 1.0}}]	[0.5574026990750067, 0.22991618489252388, 0.15107691198427015, 0.06160420404819938]	f	0.17450029102444542	[1, 2, 3, 4]	\N	[0.5574026990750066, 0.22991618489252383, 0.15107691198427012, 0.061604204048199364]
587	test3	[{"name": "s1", "comparisons": {"c1": 1.0, "c2": 2.0, "c3": 6.0, "c4": 3.0}}, {"name": "s2", "comparisons": {"c1": 0.5, "c2": 1.0, "c3": 3.0, "c4": 2.0}}, {"name": "s3", "comparisons": {"c1": 0.16666666666666666, "c2": 0.3333333333333333, "c3": 1.0, "c4": 5.0}}, {"name": "s4", "comparisons": {"c1": 0.3333333333333333, "c2": 0.5, "c3": 0.2, "c4": 1.0}}]	[0.48920414875917173, 0.25417849879136223, 0.16772184449918304, 0.08889550795028295]	f	0.1724119921765265	[1, 2, 3, 4]	\N	[0.48920414875917173, 0.25417849879136223, 0.16772184449918304, 0.08889550795028295]
\.


--
-- Data for Name: product; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.product (id, criteria, comparisons, weights, consistent, consistency_ratio, ranking, created_at, topsis_ranking, ideal_solution, negative_ideal_solution, relative_closeness) FROM stdin;
323	["test1", "test2", "test3"]	[{"c1c2": 2.0, "c1c3": 3.0}, {"c2c1": 0.5, "c2c3": 4.0}, {"c3c1": 0.3333333333333333, "c3c2": 0.25}]	[0.517133618631918, 0.3585604248241821, 0.12430595654389984]	t	0.048146131185257	[1, 2, 3]	2024-04-20 13:10:26.067716	[3, 2, 1]	[0.44325738739878684, 0.3187203776214952, 0.09751361508817878]	[0.14775246246626228, 0.0398400472026869, 0.024378403772044695]	[0.9437786943707023, 0.3672029914504877, 0.0]
324	["test1", "test2", "test3"]	[{"c1c2": 2.0, "c1c3": 3.0}, {"c2c1": 0.5, "c2c3": 4.0}, {"c3c1": 0.3333333333333333, "c3c2": 0.25}]	[0.517133618631918, 0.3585604248241821, 0.12430595654389984]	t	0.048146131185257	[1, 2, 3]	2024-04-20 13:18:58.802899	[3, 2, 1]	[0.44325738739878684, 0.3187203776214952, 0.09751361508817878]	[0.14775246246626228, 0.0398400472026869, 0.024378403772044695]	[0.9437786943707023, 0.3672029914504877, 0.0]
\.


--
-- Data for Name: users; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.users (id, fullname, username, password, email) FROM stdin;
2	oussama Raji	oussama	scrypt:32768:8:1$e2SSUbMGx9zUpkrK$7ee504cdbabeefccfab6045d2782031b01a41376241908c5bad2d34c0c4153721aceac95eba336ad69d83ae4ed9d0981e88b07ec66b474d8a39222c8745a4cb6	oussamaraji39@gmail.com
\.


--
-- Name: criteria_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.criteria_id_seq', 587, true);


--
-- Name: product_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.product_id_seq', 324, true);


--
-- Name: users_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.users_id_seq', 2, true);


--
-- Name: criteria criteria_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.criteria
    ADD CONSTRAINT criteria_pkey PRIMARY KEY (id);


--
-- Name: product product_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.product
    ADD CONSTRAINT product_pkey PRIMARY KEY (id);


--
-- Name: users users_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.users
    ADD CONSTRAINT users_pkey PRIMARY KEY (id);


--
-- PostgreSQL database dump complete
--

