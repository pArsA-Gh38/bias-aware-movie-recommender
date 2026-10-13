"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { type FormEvent, useEffect, useRef, useState } from "react";
import { ApiError, api } from "@/lib/api/client";
import type {
  RecommendationConfigurationKey,
  RecommendationResponse,
} from "@/lib/api/types";
import { RecommendationList } from "./RecommendationList";

const CONFIGURATIONS: ReadonlyArray<{
  key: RecommendationConfigurationKey;
  label: string;
  description: string;
}> = [
  {
    key: "A0_L1",
    label: "Relevance first (default)",
    description: "Prioritizes the model's relevance ranking without a diversity re-ranking adjustment.",
  },
  {
    key: "A1_L07",
    label: "Diversity — λ 0.7",
    description: "Balances relevance with more variety across the ranked results.",
  },
  {
    key: "A1_L06",
    label: "Stronger diversity — λ 0.6",
    description: "Places more emphasis on variety while continuing to account for relevance.",
  },
];

type Failure = "cold-start" | "rejected" | "unavailable" | "unexpected";

function optionLabel(key: RecommendationConfigurationKey): string {
  return CONFIGURATIONS.find((configuration) => configuration.key === key)?.label ?? key;
}

export function RecommendationsPageClient() {
  const router = useRouter();
  const [selectedConfigurationKey, setSelectedConfigurationKey] =
    useState<RecommendationConfigurationKey>("A0_L1");
  const [result, setResult] = useState<RecommendationResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [failure, setFailure] = useState<Failure | null>(null);
  const requestInFlightRef = useRef(false);
  const outcomeHeadingRef = useRef<HTMLHeadingElement>(null);

  useEffect(() => {
    if (!loading && (result || failure)) outcomeHeadingRef.current?.focus();
  }, [failure, loading, result]);

  async function generate() {
    if (requestInFlightRef.current) return;
    requestInFlightRef.current = true;
    const requestedConfigurationKey = selectedConfigurationKey;
    setLoading(true);
    setResult(null);
    setFailure(null);
    try {
      setResult(await api.recommendations({
        configuration_key: requestedConfigurationKey,
      }));
    } catch (caught) {
      if (caught instanceof ApiError && caught.status === 401) {
        router.replace("/login");
      } else if (caught instanceof ApiError && caught.status === 409) {
        setFailure("cold-start");
      } else if (caught instanceof ApiError && caught.status === 422) {
        setFailure("rejected");
      } else if (caught instanceof ApiError && (caught.status === 503 || caught.status === 504)) {
        setFailure("unavailable");
      } else {
        setFailure("unexpected");
      }
    } finally {
      setLoading(false);
      requestInFlightRef.current = false;
    }
  }

  function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    void generate();
  }

  const stale = result !== null && selectedConfigurationKey !== result.configuration_key;

  return <div className="recommendation-page">
    <header className="recommendation-page__header">
      <p className="recommendation-page__eyebrow">Personalized for your ratings</p>
      <h1>Recommendations</h1>
      <p>Choose how the service should balance relevance and variety, then generate a ranked list.</p>
    </header>

    <form className="recommendation-page__form" onSubmit={submit}>
      <fieldset className="recommendation-page__fieldset" disabled={loading}>
        <legend>Recommendation style</legend>
        <div className="recommendation-page__options">
          {CONFIGURATIONS.map((configuration) => {
            const descriptionId = `recommendation-description-${configuration.key}`;
            const inputId = `recommendation-configuration-${configuration.key}`;
            return <label className="recommendation-page__option" htmlFor={inputId} key={configuration.key}>
              <input
                id={inputId}
                type="radio"
                name="recommendation-configuration"
                value={configuration.key}
                checked={selectedConfigurationKey === configuration.key}
                aria-describedby={descriptionId}
                onChange={() => setSelectedConfigurationKey(configuration.key)}
              />
              <span className="recommendation-page__option-copy">
                <strong>{configuration.label}</strong>
                <span id={descriptionId}>{configuration.description}</span>
              </span>
            </label>;
          })}
        </div>
      </fieldset>
      <button className="recommendation-page__submit" type="submit" disabled={loading}>
        {loading ? "Generating…" : "Generate recommendations"}
      </button>
    </form>

    <section className="recommendation-page__results" aria-busy={loading} aria-label="Recommendation request and results">
      {loading && <div className="recommendation-page__loading" role="status" aria-live="polite">
        <span className="recommendation-page__spinner" aria-hidden="true" />
        Generating recommendations…
      </div>}

      {!loading && failure === "cold-start" && <div className="recommendation-page__state recommendation-page__state--guidance">
        <h2 ref={outcomeHeadingRef} tabIndex={-1}>Rate a few more movies</h2>
        <p>At least five ratings of 4 or 5 are required.</p>
        <Link href="/movies">Continue rating movies</Link>
      </div>}

      {!loading && failure && failure !== "cold-start" && <div className="recommendation-page__state recommendation-page__state--error" role="alert">
        <h2 ref={outcomeHeadingRef} tabIndex={-1}>Recommendations could not be generated</h2>
        <p>{failure === "rejected"
          ? "The selected recommendation style was rejected. Choose one of the available styles and try again."
          : failure === "unavailable"
            ? "The recommendation service is temporarily unavailable. Please try again."
            : "Could not generate recommendations. Please try again."}</p>
        <button className="recommendation-page__retry" type="button" onClick={() => void generate()}>Try again</button>
      </div>}

      {!loading && result && <div className="recommendation-page__outcome">
        <h2 ref={outcomeHeadingRef} tabIndex={-1}>
          {result.items.length === 0 ? "Recommendation status" : "Your ranked recommendations"}
        </h2>
        {stale && <p className="recommendation-page__stale" role="status" aria-live="polite">
          These results use {optionLabel(result.configuration_key)}. Generate again to apply {optionLabel(selectedConfigurationKey)}.
        </p>}
        <RecommendationList result={result} />
      </div>}
    </section>
  </div>;
}
