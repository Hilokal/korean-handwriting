import { useI18n } from "../i18n";
import { POSTS } from "../routes";
import Demo from "./Demo";

// The root page: a short lede, the live demo, and a pointer to the post.

export default function Home({
  onModelLoaded,
}: {
  onModelLoaded?: (version: string) => void;
}) {
  const { s } = useI18n();
  const [main, departures] = POSTS;
  return (
    <div className="home">
      <h1 className="home-title">{s.homeTitle}</h1>
      <p className="lede">{s.homeLede}</p>
      <Demo onModelLoaded={onModelLoaded} />
      <section className="read-more">
        <h2>{s.homeWhyTitle}</h2>
        <p>{s.homeWhyBlurb}</p>
        <p className="read-more-links">
          <a href={main.path}>{s.homeReadPost}</a>
          <a href={departures.path}>{s.homeReadDepartures}</a>
        </p>
      </section>
    </div>
  );
}
